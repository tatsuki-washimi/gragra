"""Unified verification runner for Gen1 Wave1--3 analytic catalogue.

Executes all CAS algebraic checks and numerical evaluation suites:
- AC3 (Acoustic Specialization): SymPy CAS checks + multi-precision
  candidate-vs-oracle evaluations
- T4 (Deterministic Temperature Specialization): SymPy CAS checks +
  multi-precision candidate-vs-oracle evaluations
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

SCRIPTS = [
    ("AC3 CAS checks", HERE / "ac3_cas_checks.py"),
    ("AC3 numerical evaluation", HERE / "ac3_evaluation.py"),
    ("T4 CAS checks", HERE / "t4_cas_checks.py"),
    ("T4 numerical evaluation", HERE / "t4_evaluation.py"),
]


def main() -> int:
    print("=" * 70)
    print("Running Gen1 Wave1--3 Unified Verification Suite")
    print("=" * 70)

    for name, script_path in SCRIPTS:
        print(f"\n>>> Running {name} ({script_path.name}) ...")
        res = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(HERE),
            capture_output=False,
        )
        if res.returncode != 0:
            print(f"FAILED: {name} exited with code {res.returncode}", file=sys.stderr)
            return res.returncode
        print(f">>> {name} PASSED.")

    print("\n" + "=" * 70)
    print("ALL GEN1 WAVE1--3 VERIFICATIONS COMPLETED SUCCESSFULLY (EXIT 0)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
