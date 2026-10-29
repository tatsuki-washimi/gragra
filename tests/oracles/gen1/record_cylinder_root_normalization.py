"""Additive Cylinder root/normalization evidence; no gravitational field replay.

Only the pinned original 50/80-digit Bessel root and one-dimensional radial
normalization calculations are rerun, once per unique frozen mode/radius key.
This is supplemental observed arithmetic, not a d43 callback capture or bound.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path

import mpmath as mp

SPEC = importlib.util.spec_from_file_location(
    "independent_cylinder_root_audit",
    Path(__file__).with_name("audit_frozen_refinement_traces.py"),
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def capture(value):
    return {"mpf": list(value._mpf_), "decimal": mp.nstr(value, 100)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l6-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    AUDIT.require(
        not args.output.resolve().is_relative_to(Path(__file__).resolve().parents[3]),
        "supplement must remain outside Git",
    )
    AUDIT.check_oracle_pins()
    AUDIT.require(not AUDIT.production_modules(), "production already imported")
    reference = args.l6_root / AUDIT.DATASETS["cylinder"] / "references.jsonl"
    AUDIT.require(
        AUDIT.sha256(reference) == AUDIT.EXPECTED_REFERENCE_SHA256["cylinder"],
        "frozen Cylinder reference changed",
    )
    rows = AUDIT.read_rows(reference)
    AUDIT.require(len(rows) == 576, "Cylinder fixture coverage changed")
    keys = {}
    for index, row in enumerate(rows):
        fixture = row["fixture"]
        key = (
            abs(fixture["order"]),
            fixture["radial_index"],
            fixture["radius_m"],
            fixture["axial_index"],
        )
        keys.setdefault(key, []).append({"source_row_index": index, "fixture": fixture})
    original = AUDIT.oracle("surface_reference")
    entries = []
    for key, bindings in sorted(keys.items()):
        high = original.cylinder_radial_data(*key, 80)
        low = original.cylinder_radial_data(*key, 50)
        with mp.workdps(80):
            root, normalization = high
            low_root, low_normalization = low
            estimate = (
                abs(root - low_root) * (1 + abs(root))
                + abs(normalization - low_normalization) / normalization
            )
            AUDIT.require(
                root >= 0 and normalization > 0 and mp.isfinite(estimate),
                "invalid root/normalization arithmetic",
            )
            for binding in bindings:
                metadata = rows[binding["source_row_index"]]["metadata"]
                AUDIT.require(
                    metadata["root"] == mp.nstr(root, 80)
                    and metadata["normalization"] == mp.nstr(normalization, 80),
                    "supplement high root/normalization differs from frozen metadata",
                )
            entries.append(
                {
                    "key": list(key),
                    "source_rows": bindings,
                    "at_80_digits": [capture(v) for v in high],
                    "at_50_digits": [capture(v) for v in low],
                    "root_precision_estimate": mp.nstr(estimate, 100),
                }
            )
    AUDIT.require(not AUDIT.production_modules(), "production imported")
    output = {
        "schema": "gragra.gen1.cylinder-root-normalization-supplement.v1",
        "classification": "observed supplemental root-only reconstruction",
        "method": "pinned cylinder_radial_data at 50/80; estimate at 80 digits",
        "generator_sha256": AUDIT.sha256(__file__),
        "audit_helper_sha256": AUDIT.sha256(AUDIT.__file__),
        "oracle_sha256": AUDIT.EXPECTED_ORACLE_SHA256,
        "reference_sha256": AUDIT.EXPECTED_REFERENCE_SHA256["cylinder"],
        "original_values_uncertainty_labels_unchanged": True,
        "original_d43_callback_capture": False,
        "finite_precision_or_total_error_bound_claim": False,
        "fixture_count": len(rows),
        "unique_key_count": len(entries),
        "entries": entries,
        "production_modules": AUDIT.production_modules(),
        "command": sys.argv,
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "conda": os.environ.get("CONDA_DEFAULT_ENV"),
            "mpmath": mp.__version__,
            "platform": platform.platform(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "fixture_count": len(rows),
                "unique_key_count": len(entries),
                "production_modules": AUDIT.production_modules(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
