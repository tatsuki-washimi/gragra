"""SPECFEM pilot scalar-invariant golden generator.

Reads a canonical static GLL HDF5 export of the anisotropic pilot
(``docs/validation/specfem_pilot_checks.md``),
``docs/validation/specfem_pilot_checks.md``) and records scalar invariants to
``tests/oracles/data/specfem_pilot_s1_v1.json`` so that real-pilot-derived
invariants stay under version control and remain available for regression
checks (they would otherwise live only in a transient local HDF5 that is
never committed).

The invariants are *recomputed from the datasets* (volumes/masses summed in
element_id-ascending canonical order, per
``docs/design/solver_import_contract.md`` の 4 スカラー検証規則), not copied
from ``/metadata`` -- so a later regression in either the exporter's dataset
path or its scalar path shows up as a golden mismatch.

Usage::

    # (re)generate, pointing at a freshly exported pilot static artifact
    python tests/oracles/generate_specfem_pilot_invariants.py \
        --h5 /tmp/s1_artifacts/pilot_static.h5

    # check an artifact against the committed golden (exits non-zero on
    # mismatch)
    python tests/oracles/generate_specfem_pilot_invariants.py \
        --h5 /tmp/s1_artifacts/pilot_static.h5 --check

The committed JSON is a versioned scientific oracle rather than a build
artifact; its generation script is kept beside it so the scalar invariants
can be recomputed from a supplied pilot file.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

GOLDEN_PATH = Path(__file__).parent / "data" / "specfem_pilot_s1_v1.json"

# Exact-match keys (integers / hashes / hex-encoded floats).
_EXACT_KEYS = (
    "schema_version",
    "n_elements_total",
    "ngll_shape",
    "material_ids",
    "quadrature_volume_sha256",
    "coordinates_sha256",
    "par_file_hash",
    "mesh_input_hash",
    "total_volume_m3_hex",
    "total_mass_kg_hex",
    "mass_by_material_kg_hex",
)


def _sha256(arr: np.ndarray) -> str:
    return "sha256:" + hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def compute_invariants(h5_path: Path) -> dict:
    with h5py.File(h5_path, "r") as f:
        meta = dict(f["/metadata"].attrs)
        element_id = f["/mesh/element_id"][()]
        qv = f["/mesh/quadrature_volume"][()]
        rho = f["/material/density"][()]
        material_id = f["/mesh/material_id"][()]
        coords = f["/mesh/coordinates"][()]

    order = np.argsort(element_id, kind="stable")
    qv_sorted = qv[order]
    rho_sorted = rho[order]
    mat_sorted = material_id[order]
    coords_sorted = coords[order]

    total_volume = float(np.sum(np.sum(qv_sorted, axis=1)))
    mass_per_elem = np.sum(qv_sorted * rho_sorted, axis=1)
    total_mass = float(np.sum(mass_per_elem))
    mass_by_material = {
        int(m): float(np.sum(mass_per_elem[mat_sorted == m]))
        for m in np.unique(mat_sorted)
    }

    return {
        "schema_version": int(meta["schema_version"]),
        "n_elements_total": int(element_id.size),
        "ngll_shape": [int(v) for v in meta["ngll_shape"]],
        "material_ids": [int(m) for m in np.unique(mat_sorted)],
        # Hex-encoded float64 for exact (bitwise) regression comparison;
        # the human-readable decimal fields below are informational only.
        "total_volume_m3_hex": np.float64(total_volume).tobytes().hex(),
        "total_mass_kg_hex": np.float64(total_mass).tobytes().hex(),
        "mass_by_material_kg_hex": {
            str(k): np.float64(v).tobytes().hex() for k, v in mass_by_material.items()
        },
        "total_volume_m3": total_volume,
        "total_mass_kg": total_mass,
        "mass_by_material_kg": {str(k): v for k, v in mass_by_material.items()},
        "quadrature_volume_sha256": _sha256(qv_sorted),
        "coordinates_sha256": _sha256(coords_sorted),
        "par_file_hash": str(meta["par_file_hash"]),
        "mesh_input_hash": str(meta["mesh_input_hash"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--h5", required=True, type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare against the committed golden instead of regenerating",
    )
    args = parser.parse_args()

    inv = compute_invariants(args.h5)

    if args.check:
        golden = json.loads(GOLDEN_PATH.read_text())["invariants"]
        failures = [
            f"{key}: golden={golden[key]!r} recomputed={inv[key]!r}"
            for key in _EXACT_KEYS
            if golden[key] != inv[key]
        ]
        if failures:
            print("GOLDEN MISMATCH:\n  " + "\n  ".join(failures), file=sys.stderr)
            return 1
        print(f"golden check OK ({len(_EXACT_KEYS)} exact keys) vs {GOLDEN_PATH}")
        return 0

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    payload = {
        "generator": "tests/oracles/generate_specfem_pilot_invariants.py",
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(
            timespec="seconds"
        ),
        "gragra_commit": commit,
        "source_h5": str(args.h5),
        "pilot": "specfem3d-cpu cc2e9ffa, MESH-s1-aniso (y*1.25, z*0.8), NPROC=4",
        "invariants": inv,
    }
    GOLDEN_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {GOLDEN_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
