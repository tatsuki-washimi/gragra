"""Run the oracle generator and bind its output to exact executed sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import runpy
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HASH_INPUTS = (
    "src/gragra/analytic/wave456_reference.py",
    "src/gragra/observables/wave456_reference.py",
    "tests/oracles/wave456/assumptions.yaml",
    "tests/oracles/wave456/generate_references.py",
    "tests/oracles/wave456/run_generation.py",
    "tests/oracles/wave456/verify_derivations.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except ImportError:
        return None
    return str(getattr(module, "__version__", "unknown"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.output_dir / "generation-manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite prior manifest: {manifest_path}")
    started = datetime.now(UTC)
    monotonic_start = time.monotonic()
    executed_argv = [
        str(ROOT / "tests/oracles/wave456/generate_references.py"),
        "--output-dir",
        str(args.output_dir),
    ]
    manifest = {
        "schema_version": 1,
        "base_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "runner_argv": list(sys.argv),
        "executed_generator_argv": executed_argv,
        "working_directory": str(Path.cwd()),
        "started_utc": started.isoformat(),
        "environment": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "numpy": np.__version__,
            "numba": _version("numba"),
            "h5py": _version("h5py"),
            "scipy": _version("scipy"),
            "mpmath": _version("mpmath"),
            "sympy": _version("sympy"),
        },
        "source_sha256": {
            relative: _sha256(ROOT / relative) for relative in sorted(HASH_INPUTS)
        },
    }
    exit_code = 1
    status = "FAILED"
    error = None
    prior_argv = sys.argv
    try:
        sys.argv = executed_argv
        runpy.run_path(executed_argv[0], run_name="__main__")
        exit_code = 0
        status = "COMPLETED"
    except BaseException as caught:
        error = f"{type(caught).__name__}: {caught}"
        raise
    finally:
        sys.argv = prior_argv
        ended = datetime.now(UTC)
        output_path = args.output_dir / "reference-values.json"
        manifest.update(
            {
                "ended_utc": ended.isoformat(),
                "duration_s": time.monotonic() - monotonic_start,
                "exit_state": {
                    "status": status,
                    "exit_code": exit_code,
                    "error": error,
                },
                "output": {
                    "path": str(output_path),
                    "sha256": _sha256(output_path) if output_path.exists() else None,
                },
            }
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
