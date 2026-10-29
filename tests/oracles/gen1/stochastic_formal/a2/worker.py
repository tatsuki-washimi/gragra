"""Reference worker: immutable supplied fixture/procedure bytes, no production."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from oracle import independence, records, scalar_record
from review import inventory


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--proposal", type=Path, required=True)
    p.add_argument("--calibration", action="store_true")
    p.add_argument("--authorization", type=Path)
    p.add_argument("--inspect-environment", action="store_true")
    args = p.parse_args()
    independence()
    from runner import verify_authorization

    if args.inspect_environment:
        print(json.dumps(inventory(), sort_keys=True))
        return 0
    policy = json.loads(args.proposal.read_bytes())
    procedure_hash = hashlib.sha256(args.proposal.read_bytes()).hexdigest()
    if not args.calibration:
        if args.authorization is None:
            p.error("formal worker HOLD: reviewed root F/E authorization required")
        auth = json.loads(args.authorization.read_bytes())
        verify_authorization(auth, args.proposal)
        procedure_hash = auth["procedure_sha256"]
    for fixture in policy["calibration"] if args.calibration else policy["fixtures"]:
        if (
            not isinstance(fixture, dict)
            or fixture.get("pressure_convention") != "surface_total"
            or fixture.get("psd_convention") != "surface_total_one_sided_per_hz"
        ):
            raise ValueError("exact surface-total pressure convention required")
    before = inventory()
    if not args.calibration and before != auth["reference_runtime_inventory"]:
        raise ValueError("before-compute runtime origin/byte closure mismatch")
    if args.calibration:
        for fixture in policy["calibration"]:
            x = fixture["nominal_exact_x"] if isinstance(fixture, dict) else fixture
            from decimal import Decimal

            if Decimal(x) not in {Decimal(v) for v in (".0023", ".41", "2.7", "47")}:
                p.error("only frozen disjoint calibration x permitted")
            fixture = {
                **fixture,
                "key_base": {
                    "family": "A2",
                    "fixture_id": fixture["fixture_id"],
                    "frequency_hz": fixture["inputs"]["frequency_hz"],
                    "target_id": "nominal_x",
                    "observable": "isotropic_depth_factor",
                    "set": "auxiliary",
                },
            }
            row, trace = scalar_record(fixture, procedure_hash, policy["source_pins"])
            print(json.dumps(row, sort_keys=True, separators=(",", ":")), flush=True)
            sys.stderr.write(
                json.dumps({"calibration_x": x, "trace": trace}, sort_keys=True) + "\n"
            )
        if inventory() != before:
            raise ValueError("after-compute runtime origin/byte closure mismatch")
        return 0
    for fixture in policy["fixtures"]:
        rows, trace = records(fixture, procedure_hash, policy["source_pins"])
        scalar, nominal = scalar_record(fixture, procedure_hash, policy["source_pins"])
        rows.append(scalar)
        trace["nominal_exact_x_reference"] = nominal
        for row in rows:
            print(json.dumps(row, sort_keys=True, separators=(",", ":")), flush=True)
        sys.stderr.write(
            json.dumps(
                {"fixture_id": fixture["key_base"]["fixture_id"], "trace": trace},
                sort_keys=True,
            )
            + "\n"
        )
        sys.stderr.flush()
    if inventory() != before:
        raise ValueError("after-compute runtime origin/byte closure mismatch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
