"""Disambiguate Cartesian and toroidal centroids in preserved transport data."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from scripts.research_wave5_environment_transport import (
    CASE_SPECS,
    TIMES_S,
    _parse_native_field,
    _parse_native_scalar_field,
    _sha256,
    periodic_center_of_mass,
)


def cartesian_signed_mass_centroid(
    positions_m: np.ndarray, signed_mass_kg: np.ndarray
) -> np.ndarray:
    """Return the ordinary Cartesian first moment divided by signed mass."""
    positions = np.asarray(positions_m, dtype=np.float64)
    mass = np.asarray(signed_mass_kg, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions_m must have shape (N,3)")
    if mass.shape != (len(positions),):
        raise ValueError("signed_mass_kg must have shape (N,)")
    if not np.isfinite(positions).all() or not np.isfinite(mass).all():
        raise ValueError("centroid inputs must be finite")
    total = float(np.sum(mass))
    if total == 0.0:
        raise ValueError("signed total mass must be nonzero")
    result = np.asarray(np.sum(positions * mass[:, None], axis=0) / total)
    result.setflags(write=False)
    return result


def run_supplement(
    *, output: Path, transport_dir: Path, parent_verification: Path
) -> None:
    output = output.resolve()
    transport_dir = transport_dir.resolve()
    parent_verification = parent_verification.resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest_path = transport_dir / "transport-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("transport manifest is not complete")
    cases = {}
    for spec in CASE_SPECS:
        case_dir = transport_dir / spec.name
        result_path = transport_dir / f"{spec.name}-result.json"
        stored = json.loads(result_path.read_text(encoding="utf-8"))
        time_records = {}
        for time_s in TIMES_S:
            time_name = format(time_s, "g")
            count = spec.level**3
            positions = _parse_native_field(
                case_dir / time_name / "C",
                object_name="C",
                components=3,
                expected_count=count,
                expected_dimensions=(0, 1, 0, 0, 0, 0, 0),
            ).values
            volumes = _parse_native_scalar_field(
                case_dir / time_name / "Vc",
                object_name="Vc",
                expected_count=count,
                expected_dimensions=(0, 3, 0, 0, 0, 0, 0),
            )
            density = _parse_native_scalar_field(
                case_dir / time_name / "delta_rho",
                object_name="delta_rho",
                expected_count=count,
                expected_dimensions=(1, -3, 0, 0, 0, 0, 0),
            )
            signed_mass = density * volumes
            circular = periodic_center_of_mass(positions, signed_mass)
            cartesian = cartesian_signed_mass_centroid(positions, signed_mass)
            legacy = np.asarray(
                stored["science"]["times"][time_name]["native"]["center_of_mass_m"]
            )
            if not np.array_equal(circular, legacy):
                raise AssertionError(
                    "stored legacy center does not match circular moment"
                )
            time_records[time_name] = {
                "signed_mass_kg": float(np.sum(signed_mass)),
                "periodic_circular_first_moment_center_m": circular.tolist(),
                "cartesian_signed_mass_centroid_m": cartesian.tolist(),
                "legacy_center_of_mass_m_interpretation": (
                    "periodic circular first trigonometric moment on a unit torus"
                ),
                "density": {
                    "path": str(case_dir / time_name / "delta_rho"),
                    "sha256": _sha256(case_dir / time_name / "delta_rho"),
                },
                "centres": {
                    "path": str(case_dir / time_name / "C"),
                    "sha256": _sha256(case_dir / time_name / "C"),
                },
                "volumes": {
                    "path": str(case_dir / time_name / "Vc"),
                    "sha256": _sha256(case_dir / time_name / "Vc"),
                },
            }
        cases[spec.name] = time_records
    result = {
        "status": "complete",
        "classification": (
            "observed recomputation from preserved native OpenFOAM fields"
        ),
        "generated_at": datetime.now(UTC).isoformat(),
        "scope": {
            "transport_boundaries": "cyclic",
            "gravity_domain": (
                "one finite cube only; no infinite periodic gravity images"
            ),
            "interpretation": (
                "circular and Cartesian centers are distinct statistics; their "
                "difference is not a material center-of-mass lag"
            ),
        },
        "execution": {
            "argv": list(sys.argv),
            "argv_shell": shlex.join(sys.argv),
        },
        "source": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256(Path(__file__).resolve()),
        },
        "dependencies": {
            "transport_manifest": {
                "path": str(manifest_path),
                "sha256": _sha256(manifest_path),
            },
            "parent_native_verification": {
                "path": str(parent_verification),
                "sha256": _sha256(parent_verification),
            },
        },
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transport-dir", type=Path, required=True)
    parser.add_argument("--parent-verification", type=Path, required=True)
    args = parser.parse_args()
    run_supplement(
        output=args.output,
        transport_dir=args.transport_dir,
        parent_verification=args.parent_verification,
    )


if __name__ == "__main__":
    main()
