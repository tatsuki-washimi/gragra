"""Generate and export the W6 manufactured OpenFOAM case without a solver."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="ascii")


def _field(name: str, values: str) -> str:
    return f"""FoamFile {{ version 2.0; format ascii; class volScalarField;
object {name}; }}
dimensions [1 -3 0 0 0 0 0];
internalField nonuniform List<scalar> 2 ({values});
boundaryField {{ walls {{ type zeroGradient; }} }}
"""


def generate_case(case: Path) -> None:
    _write(
        case / "system/blockMeshDict",
        """FoamFile { version 2.0; format ascii; class dictionary;
object blockMeshDict; }
scale 1;
vertices ((0 0 0) (2 0 0) (2 1 0) (0 1 0) (0 0 1) (2 0 1) (2 1 1) (0 1 1));
blocks (hex (0 1 2 3 4 5 6 7) (2 1 1) simpleGrading (1 1 1));
edges (); boundary (); defaultPatch { name walls; type wall; }
""",
    )
    _write(
        case / "system/controlDict",
        "FoamFile { version 2.0; format ascii; class dictionary; "
        "object controlDict; }\n"
        "application none; startFrom startTime; startTime 0; stopAt endTime; "
        "endTime .125; deltaT .125; writeControl timeStep; writeInterval 1;\n",
    )
    for time, delta, rho in (
        ("0", "-1 2", "999 1002"),
        ("0.125", "2 -1", "1002 999"),
    ):
        _write(case / time / "delta_rho", _field("delta_rho", delta))
        _write(case / time / "rho", _field("rho", rho))


def _run(activation: Path, argv: list[str], log: Path) -> int:
    command = [
        "bash",
        "-lc",
        'source "$1" && shift && exec "$@"',
        "bash",
        str(activation),
        *argv,
    ]
    with log.open("a", encoding="utf-8") as stream:
        stream.write("argv=" + json.dumps(argv) + "\n")
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        stream.write(f"exit={result.returncode}\n")
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation-script", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output, activation = args.output_dir.resolve(), args.activation_script.resolve()
    case = output / "case"
    if output.exists() or not activation.is_file():
        raise SystemExit("output exists or activation script is absent")
    generate_case(case)
    log = output / "utilities.log"
    commands = [
        ["blockMesh", "-case", str(case)],
        ["checkMesh", "-case", str(case), "-time", "0,.125"],
        [
            "foamPostProcess",
            "-case",
            str(case),
            "-time",
            "0,.125",
            "-func",
            "writeCellCentres",
        ],
        [
            "foamPostProcess",
            "-case",
            str(case),
            "-time",
            "0,.125",
            "-func",
            "writeCellVolumes",
        ],
        [
            "foamToVTK",
            "-case",
            str(case),
            "-time",
            "0,.125",
            "-useTimeName",
            "-ascii",
            "-noPointValues",
            "-polyhedra",
            "none",
            "-fields",
            "(rho delta_rho C Vc)",
        ],
    ]
    statuses = []
    for argv in commands:
        status = _run(activation, argv, log)
        statuses.append({"argv": argv, "exit": status})
        if status:
            raise SystemExit(status)
    hashes = {
        str(path.relative_to(case)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(case.rglob("*"))
        if path.is_file()
    }
    (output / "manifest.json").write_text(
        json.dumps({"commands": statuses, "input_sha256": hashes}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
