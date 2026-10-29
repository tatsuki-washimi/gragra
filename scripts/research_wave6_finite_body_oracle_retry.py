#!/usr/bin/env python3
"""Retry only unresolved Wave 6 A-oracle calls with larger fixed budgets."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from gragra.constants import G_SI
from scripts.research_wave4_finite_body import (
    OBSERVABLES,
    _array_record,
    normalized_error,
)
from scripts.research_wave6_finite_body_cavity import (
    _independent_oracle,
    _load_oracle_module,
    add_fixed_quadrature_error_curves,
    wave6_case_matrix,
)

STRICT_RTOL = 1e-11


def _convergence_dict(value) -> dict:
    names = (
        "status",
        "normalized_difference",
        "low_order",
        "high_order",
        "subbox_count",
        "max_depth_reached",
        "evaluations",
    )
    return {name: getattr(value, name) for name in names}


def retry_unresolved_oracle(
    module,
    host_bounds,
    wavelength,
    *,
    initial_convergence: dict,
    retry_budgets=(8192, 16384),
):
    """Preserve the 4096 attempt and retry without relaxing the 1e-11 gate."""
    attempts = [
        {
            "max_evaluations": 4096,
            "convergence": dict(initial_convergence),
        }
    ]
    last_value = None
    last_convergence = None
    for budget in retry_budgets:
        last_value, last_convergence = _independent_oracle(
            module,
            host_bounds,
            wavelength,
            max_evaluations=int(budget),
        )
        attempts.append(
            {
                "max_evaluations": int(budget),
                "convergence": _convergence_dict(last_convergence),
            }
        )
        if (
            last_convergence.status == "CONVERGED"
            and last_convergence.normalized_difference <= STRICT_RTOL
        ):
            break
    return last_value, last_convergence, attempts


def _decode_array_record(record: dict) -> np.ndarray:
    values = np.asarray(record["values"], dtype=np.float64)
    if record["encoding"] == "real_imag_last_axis":
        result = values[..., 0] + 1j * values[..., 1]
        return np.asarray(result, dtype=np.complex128)
    if record["encoding"] != "plain":
        raise ValueError(f"unknown array encoding {record['encoding']!r}")
    return values


def finalize_wave6_oracles(
    result_path: Path,
    oracle_module_path: Path,
    accepted_sha256: str,
    *,
    retry_budgets=(8192, 16384),
) -> Path:
    """Update only A-reference fields in an existing complete 15-case result."""
    original_bytes = result_path.read_bytes()
    original_sha256 = hashlib.sha256(original_bytes).hexdigest()
    result = json.loads(original_bytes)
    expected_ids = {item["case_id"] for item in wave6_case_matrix()}
    observed_ids = {item["case_id"] for item in result["cases"]}
    if observed_ids != expected_ids or len(result["cases"]) != len(expected_ids):
        raise ValueError("resume result is not the complete fixed 15-case matrix")
    module = _load_oracle_module(oracle_module_path)
    if module.__source_sha256__ != accepted_sha256:
        raise ValueError("A module bytes do not match the parent-accepted SHA-256")
    if result["oracle_module"]["sha256"] != accepted_sha256:
        raise ValueError("existing production result used different A module bytes")

    unresolved_ids = []
    for case in result["cases"]:
        initial = dict(case["A_convergence"])
        case.setdefault(
            "A_attempts",
            [{"max_evaluations": 4096, "convergence": initial}],
        )
        if case["A_numerical_status"] == "ORACLE_UNRESOLVED":
            oracle, convergence, attempts = retry_unresolved_oracle(
                module,
                np.asarray(case["host_bounds_m"], dtype=np.float64),
                float(case["wavelength_m"]),
                initial_convergence=initial,
                retry_budgets=retry_budgets,
            )
            case["A_attempts"] = attempts
            case["A_convergence"] = _convergence_dict(convergence)
            converged = (
                convergence.status == "CONVERGED"
                and convergence.normalized_difference <= STRICT_RTOL
            )
            if converged:
                for name in OBSERVABLES:
                    reference_physical = G_SI * getattr(oracle, name)
                    observable = case["observables"][name]
                    mass = _decode_array_record(observable["mass_displacement"])
                    bulk_interface = _decode_array_record(
                        observable["bulk_plus_interface"]
                    )
                    observable["independent_A"] = _array_record(reference_physical)
                    observable["mass_vs_A_normalized_error"] = normalized_error(
                        mass, reference_physical, name
                    ).tolist()
                    observable["bulk_interface_vs_A_normalized_error"] = (
                        normalized_error(
                            bulk_interface, reference_physical, name
                        ).tolist()
                    )
                case["A_numerical_status"] = "CONVERGED_ACCEPTED"
            else:
                case["A_numerical_status"] = "ORACLE_UNRESOLVED"
                unresolved_ids.append(case["case_id"])
        else:
            case["A_numerical_status"] = "CONVERGED_ACCEPTED"
        case["A_parent_acceptance"] = "accepted_exact_bytes"

    result["oracle_module"]["parent_acceptance"] = "accepted_exact_bytes"
    for study in result["fixed_mesh_quadrature_studies"]:
        add_fixed_quadrature_error_curves(study)
    first_production_sha = result.get("oracle_finalization", {}).get(
        "original_result_sha256", original_sha256
    )
    result["oracle_finalization"] = {
        "classification": "Observed",
        "original_result_sha256": first_production_sha,
        "latest_input_result_sha256": original_sha256,
        "production_cases_recomputed": False,
        "fixed_mesh_quadrature_recomputed": False,
        "only_unresolved_A_oracles_retried": True,
        "strict_rtol": STRICT_RTOL,
        "retry_budgets": list(retry_budgets),
        "remaining_unresolved_case_ids": unresolved_ids,
        "accepted_A_sha256": accepted_sha256,
        "fixed_quadrature_error_curves_postprocessed": True,
        "finished_utc": datetime.now(UTC).isoformat(),
    }
    preserved = result_path.with_name(
        "wave6_finite_body_cavity_results-before-oracle-retry.json"
    )
    if not preserved.exists():
        preserved.write_bytes(original_bytes)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--oracle-module", type=Path, required=True)
    parser.add_argument("--accepted-sha256", required=True)
    parser.add_argument("--retry-budget", type=int, action="append", default=None)
    args = parser.parse_args()
    print(
        finalize_wave6_oracles(
            args.result,
            args.oracle_module,
            args.accepted_sha256,
            retry_budgets=(8192, 16384)
            if args.retry_budget is None
            else tuple(args.retry_budget),
        )
    )


if __name__ == "__main__":
    main()
