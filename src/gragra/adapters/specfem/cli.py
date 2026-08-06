"""Command-line interface for the SPECFEM3D Cartesian GLL certified exporter
(Step 6): ``export`` (the main entry point) and ``capture-run`` (the
runtime-executable-identity capture wrapper).

Usage
-----

.. code-block:: bash

    python -m gragra.adapters.specfem export \\
        --specfem-dir /path/to/run --mode static -o out.h5

    python -m gragra.adapters.specfem capture-run \\
        --manifest solver_manifest.json --role solver \\
        -- /path/to/bin/xspecfem3D

``capture-run`` calls ``provenance.capture_executable`` **immediately before**
running the given command (per that function's "capture immediately before
running" usage discipline -- a stale binary resolved after the fact cannot
be guaranteed to be the one that actually ran), writes the captured entry as
JSON to ``--manifest``, then executes the command via ``subprocess.run`` and
propagates its exit code unchanged.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from gragra.adapters.specfem import export as _export
from gragra.adapters.specfem import provenance as _provenance
from gragra.adapters.specfem import schema as _schema


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gragra.adapters.specfem",
        description="SPECFEM3D Cartesian GLL certified/smoke exporter (S1).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    export_p = sub.add_parser("export", help="Export a canonical GLL HDF5 file.")
    export_p.add_argument(
        "--specfem-dir", required=True, type=Path, help="SPECFEM3D run tree root."
    )
    export_p.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        dest="output_path",
        help="Destination canonical HDF5 path.",
    )
    export_p.add_argument(
        "--mode",
        choices=("static", "time_sampled"),
        default="static",
        dest="output_sampling_kind",
        help="'static' (mesh-only, Tier A gate) or 'time_sampled' (non-certified).",
    )
    export_p.add_argument(
        "--ngll", type=int, nargs=3, default=None, metavar=("NX", "NY", "NZ")
    )
    export_p.add_argument("--exporter-code-ref", default="unknown")
    export_p.add_argument("--solver-version", default="unknown")
    export_p.add_argument(
        "--float-precision", choices=("single", "double"), default="single"
    )
    export_p.add_argument("--density-source", default="proc_rho_bin_derived")
    export_p.add_argument("--f-resolved-estimate", type=float, default=None)
    export_p.add_argument("--t0", type=float, default=None)
    export_p.add_argument("--t0-source", default=None)

    capture_p = sub.add_parser(
        "capture-run",
        help="Capture an executable's runtime identity, then exec the given command.",
    )
    capture_p.add_argument("--manifest", required=True, type=Path)
    capture_p.add_argument("--role", required=True)
    capture_p.add_argument(
        "command", nargs=argparse.REMAINDER, help="Command to run, after '--'."
    )

    return parser


def _run_export(args: argparse.Namespace) -> int:
    ngll_shape = tuple(args.ngll) if args.ngll else _schema.EXPECTED_NGLL
    config = _export.ExportConfig(
        specfem_dir=args.specfem_dir,
        output_path=args.output_path,
        output_sampling_kind=args.output_sampling_kind,
        ngll_shape=ngll_shape,
        exporter_code_ref=args.exporter_code_ref,
        solver_version=args.solver_version,
        float_precision=args.float_precision,
        density_source=args.density_source,
        f_resolved_estimate=args.f_resolved_estimate,
        t0=args.t0,
        t0_source=args.t0_source,
    )
    out = _export.export_specfem_gll_hdf5(config)
    print(out)
    return 0


def _run_capture_run(args: argparse.Namespace) -> int:
    command = [c for c in args.command if c != "--"]
    if not command:
        print("capture-run: no command given after '--'", file=sys.stderr)
        return 2

    entry = _provenance.capture_executable(command[0], role=args.role)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n")

    completed = subprocess.run(command, check=False)
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "export":
        return _run_export(args)
    if args.command == "capture-run":
        return _run_capture_run(args)
    parser.error(f"unknown command {args.command!r}")  # pragma: no cover
    return 2  # pragma: no cover -- parser.error() above always raises SystemExit
