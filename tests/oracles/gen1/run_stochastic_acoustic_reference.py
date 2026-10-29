"""Production-blind development selfchecks. Formal generation is held closed.

This CLI intentionally provides no formal-output path before common/family
freeze and admission of a reviewed runner/sealer in an exact clean E tree.
Selfchecks use predeclared disjoint calibration coordinates only. No comparison
process, production helper, gragra import or production output is used.
"""

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import mpmath as mp

if __package__:
    from .stochastic_acoustic_reference import (
        CALIBRATION,
        acoustic_reference,
        decimal_string,
        deny_production_imports,
        depth_estimate,
        green_closed,
        green_integral,
        volume_estimate,
    )
else:
    from stochastic_acoustic_reference import (
        CALIBRATION,
        acoustic_reference,
        decimal_string,
        deny_production_imports,
        depth_estimate,
        green_closed,
        green_integral,
        volume_estimate,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selfcheck", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.formal or not args.selfcheck:
        parser.error("formal HOLD: reviewed frozen F + exact clean E + sealer required")
    repository = Path(__file__).resolve().parents[3]
    if args.output and args.output.resolve().is_relative_to(repository):
        parser.error("development raw output must remain outside Git")
    if args.output and args.output.exists():
        parser.error("preserve selfchecks; output must be a fresh path")
    with deny_production_imports(), mp.workdps(80):
        q, b, z = CALIBRATION["A1"].values()
        j, dj, tail = green_integral(q, b, z)
        exact, dexact = green_closed(*map(mp.mpf, [q, b, z]))
        green_error = max(abs(j - exact), abs(dj - dexact))
        records = {
            "A1_green": {
                "difference_estimate": decimal_string(green_error),
                "tail_bound_j": decimal_string(tail),
                "resolved": bool(green_error < mp.mpf("1e-45") + tail),
            },
            "A1_csd": acoustic_reference(
                {
                    "family": "A1",
                    "fixture_id": "development-csd",
                    "frequency_hz": "2.3",
                    "sound_speed_m_s": "331.7",
                    "pressure_psd_pa2_hz": "0.023",
                    "g_si": "6.6743e-11",
                    "target_id": "development-center",
                    "observable": "acceleration_csd",
                    "representation": "one_sided_per_hz_csd",
                    "set": "development",
                }
            ),
            "A2": {x: depth_estimate(x) for x in CALIBRATION["A2"]},
            "A3": {x: volume_estimate(x) for x in CALIBRATION["A3"]},
        }
    passed = (
        records["A1_green"]["resolved"] and records["A1_csd"]["resolved"]
    ) and all(
        record["resolved"]
        for family in ["A2", "A3"]
        for record in records[family].values()
    )
    ctx = mp.mp.clone()
    ctx.dps = 80
    payload = {
        "purpose": "independent_development_selfcheck_not_acceptance",
        "calibration_inputs": {
            "A1": {k: decimal_string(ctx.mpf(v)) for k, v in CALIBRATION["A1"].items()},
            "A2": [decimal_string(ctx.mpf(v)) for v in CALIBRATION["A2"]],
            "A3": [decimal_string(ctx.mpf(v)) for v in CALIBRATION["A3"]],
        },
        "records": records,
        "production_values_seen": False,
        "production_imported": False,
        "python": platform.python_version(),
        "mpmath": mp.__version__,
        "generator_sha256": hashlib.sha256(
            Path(__file__).with_name("stochastic_acoustic_reference.py").read_bytes()
        ).hexdigest(),
        "resolved": passed,
    }
    text = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
    sys.stdout.write(text)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
