"""Calibration-only independent T1 runner. Formal execution fails closed.

Reviewed F_T1/common schema, source-normalization review, and exact clean E
with reviewed runner/sealer have NOT been supplied. No CLI option bypasses this
hold. A future reviewed revision must implement formal admission separately.
"""

import argparse
import json
from pathlib import Path

from stochastic_temperature_reference import evaluate_calibration


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["calibration", "formal"], required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == "formal":
        parser.error("FORMAL_HOLD: reviewed F/E/sealer admission has not been supplied")
    inputs = json.loads(args.inputs.read_text(encoding="utf-8"))
    result = evaluate_calibration(inputs)
    payload = (
        json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    )
    # Fresh output only: failed/previous calibration artifacts are never overwritten.
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
