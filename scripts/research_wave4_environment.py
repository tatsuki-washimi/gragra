"""Consume the accepted F free-surface benchmark and record thermo checks."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from gragra.fields.thermodynamic import (
    adiabatic_pressure_density_perturbation,
    isobaric_temperature_density_perturbation,
)

ALPHAS_M = (-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02)
FREE_SURFACE_TARGETS_M = np.array(((0.2, -0.1, 2.5),), dtype=np.float64)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_free_surface_result(result: dict[str, object]) -> None:
    expected_alphas = list(ALPHAS_M)
    if result.get("column_bounds_m") != [
        [-1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
    ]:
        raise ValueError("free-surface column bounds differ from the approved box")
    if result.get("densities_kg_m3") != [1000.0, 1.2]:
        raise ValueError("free-surface densities differ from the approved fluids")
    if result.get("alphas_m") != expected_alphas:
        raise ValueError("free-surface alpha sweep is incomplete or reordered")
    if result.get("combination_policy") != "compare_only_do_not_add":
        raise ValueError("volume and sheet representations must not be added")
    if result.get("mass_change_zero_assumed") is not False:
        raise ValueError("signed mass change must not be assumed zero")
    controls = result.get("controls")
    if not isinstance(controls, dict) or not all(
        controls.get(name) is True
        for name in (
            "zero_displacement_exact_zero",
            "zero_density_jump_exact_zero",
            "sign_reversal_linear_sheet",
        )
    ):
        raise ValueError("free-surface zero/sign controls did not all pass")
    signed_mass = result.get("signed_mass_change_kg")
    if not isinstance(signed_mass, dict):
        raise ValueError("free-surface signed mass changes are absent")
    if not (float(signed_mass["-0.02"]) < 0.0 < float(signed_mass["0.02"])):
        raise ValueError("free-surface signed mass changes lost alpha sign")
    responses = result.get("responses")
    if not isinstance(responses, dict) or set(responses) != {
        str(value) for value in ALPHAS_M
    }:
        raise ValueError("free-surface responses do not cover the alpha matrix")


def _load_f_module(script: Path, f_worktree: Path):
    sys.path.insert(0, str(f_worktree))
    try:
        specification = importlib.util.spec_from_file_location(
            "gragra_lane_f_wave5_finite_body", script
        )
        if specification is None or specification.loader is None:
            raise ImportError(f"cannot load accepted F helper from {script}")
        module = importlib.util.module_from_spec(specification)
        sys.modules[specification.name] = module
        specification.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(f_worktree))


def run_study(
    output: Path,
    f_acceptance: Path,
    f_worktree: Path,
    f_script_sha256: str,
    f_artifact: Path,
    f_artifact_sha256: str,
    worktree: Path,
) -> None:
    output = Path(output).resolve()
    f_acceptance = Path(f_acceptance).resolve()
    f_worktree = Path(f_worktree).resolve()
    f_artifact = Path(f_artifact).resolve()
    worktree = Path(worktree).resolve()
    if output.exists():
        raise FileExistsError(output)
    acceptance = json.loads(f_acceptance.read_text(encoding="utf-8"))
    if "ACCEPT" not in str(acceptance.get("status", "")).upper():
        raise ValueError("F parent acceptance status is not accepted")
    script = f_worktree / "scripts/research_wave5_finite_body.py"
    if _sha256(script) != f_script_sha256:
        raise ValueError("accepted F helper source hash mismatch")
    if _sha256(f_artifact) != f_artifact_sha256:
        raise ValueError("accepted F artifact hash mismatch")
    source_path = Path(__import__("gragra").__file__).resolve()
    expected_source = worktree / "src/gragra/__init__.py"
    if source_path != expected_source:
        raise RuntimeError(
            f"gragra imported from {source_path}, expected {expected_source}"
        )
    stored = json.loads(f_artifact.read_text(encoding="utf-8"))["free_surface"]
    validate_free_surface_result(stored)
    module = _load_f_module(script, f_worktree)
    recomputed = module.free_surface_column_study(
        alphas_m=np.asarray(ALPHAS_M),
        targets_m=FREE_SURFACE_TARGETS_M,
        order=16,
    )
    validate_free_surface_result(recomputed)
    byte_equivalent = json.dumps(recomputed, sort_keys=True) == json.dumps(
        stored, sort_keys=True
    )
    if not byte_equivalent:
        raise AssertionError("accepted F free-surface artifact was not reproduced")
    thermo = {
        "isobaric_temperature": {
            "formula": "-rho0*delta_T/T0",
            "rho0_kg_m3": 1.2,
            "T0_k": 300.0,
            "delta_T_k": [-1.0, 0.0, 1.0],
            "delta_rho_kg_m3": isobaric_temperature_density_perturbation(
                1.2, np.array((-1.0, 0.0, 1.0)), 300.0
            ).tolist(),
        },
        "adiabatic_pressure": {
            "formula": "delta_p/c_s^2",
            "pressure_input": "physical Pa; not kinematic pressure",
            "delta_p_pa": 1.0,
            "c_s_m_s": 343.0,
            "delta_rho_kg_m3": float(
                adiabatic_pressure_density_perturbation(1.0, 343.0)
            ),
        },
    }
    output.mkdir(parents=True)
    result = {
        "status": "complete",
        "classification": {
            "free_surface": "observed numerical quadrature from accepted F helper",
            "thermodynamic": "inferred linear thermodynamic identities",
            "imputed_values": [],
        },
        "generated_at": datetime.now(UTC).isoformat(),
        "source": {
            "gragra": str(source_path),
            "script": str(Path(__file__).resolve()),
        },
        "dependencies": {
            "F_parent_acceptance": {
                "path": str(f_acceptance),
                "sha256": _sha256(f_acceptance),
            },
            "F_helper": {"path": str(script), "sha256": _sha256(script)},
            "F_artifact": {
                "path": str(f_artifact),
                "sha256": _sha256(f_artifact),
            },
        },
        "free_surface_recomputed_equal_to_accepted_artifact": byte_equivalent,
        "thermodynamic": thermo,
        "free_surface": stored,
    }
    (output / "environment-wave4-manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--f-acceptance", type=Path, required=True)
    parser.add_argument("--f-worktree", type=Path, required=True)
    parser.add_argument("--f-script-sha256", required=True)
    parser.add_argument("--f-artifact", type=Path, required=True)
    parser.add_argument("--f-artifact-sha256", required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    args = parser.parse_args()
    run_study(
        args.output_dir,
        args.f_acceptance,
        args.f_worktree,
        args.f_script_sha256,
        args.f_artifact,
        args.f_artifact_sha256,
        args.worktree,
    )


if __name__ == "__main__":
    main()
