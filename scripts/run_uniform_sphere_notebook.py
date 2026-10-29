"""Execute the public uniform-sphere notebook in this Python environment."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import nbformat
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

REPOSITORY = Path(__file__).resolve().parents[1]
NOTEBOOK = REPOSITORY / "examples" / "uniform_sphere_workflow.ipynb"
DEFAULT_OUTPUT = Path.home() / ".cache" / "gragra" / "uniform-sphere-workflow"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="external directory for the executed notebook, plot, and provenance",
    )
    return parser


def execute_notebook(output_dir: Path) -> dict[str, Path]:
    """Execute with a temporary kernelspec pinned to the running interpreter."""
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not NOTEBOOK.is_file():
        raise FileNotFoundError(f"notebook not found: {NOTEBOOK}")

    executed_path = output_dir / "uniform_sphere_workflow.executed.ipynb"
    plot_path = output_dir / "uniform_sphere_workflow.png"
    provenance_path = output_dir / "provenance.json"
    notebook = nbformat.read(NOTEBOOK, as_version=4)

    with tempfile.TemporaryDirectory(prefix="gragra-kernel-") as temporary:
        root = Path(temporary)
        kernel_name = "gragra-workflow-" + str(os.getpid())
        spec_dir = root / kernel_name
        spec_dir.mkdir()
        kernel_json = {
            "argv": [
                sys.executable,
                "-m",
                "ipykernel_launcher",
                "-f",
                "{connection_file}",
            ],
            "display_name": f"gragra workflow ({sys.executable})",
            "language": "python",
            "metadata": {"debugger": True},
            "env": {"GRAGRA_WORKFLOW_PLOT": str(plot_path)},
        }
        (spec_dir / "kernel.json").write_text(
            json.dumps(kernel_json, indent=2) + "\n", encoding="utf-8"
        )
        kernel_specs = KernelSpecManager(kernel_dirs=[str(root)])
        if kernel_name not in kernel_specs.find_kernel_specs():
            raise RuntimeError("temporary interpreter-pinned kernelspec was not found")
        notebook.metadata.kernelspec = {
            "display_name": kernel_json["display_name"],
            "language": "python",
            "name": kernel_name,
        }
        from jupyter_client import KernelManager

        kernel_manager = KernelManager(
            kernel_name=kernel_name,
            kernel_spec_manager=kernel_specs,
        )
        client = NotebookClient(
            notebook,
            km=kernel_manager,
            timeout=600,
            kernel_name=kernel_name,
            resources={"metadata": {"path": str(REPOSITORY)}},
            allow_errors=False,
        )
        client.execute(cleanup_kc=True)

    nbformat.write(notebook, executed_path)
    provenance = _read_and_validate_provenance(notebook)
    provenance["runner_executable"] = str(Path(sys.executable).resolve())
    provenance["runner_executable_as_invoked"] = sys.executable
    provenance["runner_prefix"] = str(Path(sys.prefix).resolve())
    provenance["notebook"] = str(NOTEBOOK)
    provenance["executed_notebook"] = str(executed_path)
    provenance["plot"] = str(plot_path)
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not plot_path.is_file():
        raise RuntimeError(f"notebook did not create its requested plot: {plot_path}")
    return {
        "notebook": executed_path,
        "plot": plot_path,
        "provenance": provenance_path,
    }


def _read_and_validate_provenance(notebook: nbformat.NotebookNode) -> dict[str, str]:
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            raw = data.get("application/json", data)
            if isinstance(raw, dict) and "python_executable" in raw:
                reported = Path(raw["python_executable"]).resolve()
                expected = Path(sys.executable).resolve()
                if reported != expected:
                    raise RuntimeError(
                        "kernel executable mismatch: "
                        f"runner resolves to {expected}, kernel reported {reported}"
                    )
                kernel_prefix = Path(raw.get("python_prefix", "")).resolve()
                runner_prefix = Path(sys.prefix).resolve()
                if kernel_prefix != runner_prefix:
                    raise RuntimeError(
                        "kernel environment prefix mismatch: "
                        f"expected {sys.prefix}, got {raw.get('python_prefix')}"
                    )
                import_path = Path(raw["gragra_import_path"]).resolve()
                expected_package = (REPOSITORY / "src" / "gragra").resolve()
                if import_path != expected_package:
                    raise RuntimeError(
                        "kernel imported gragra outside this checkout: "
                        f"expected {expected_package}, got {import_path}"
                    )
                return {key: str(value) for key, value in raw.items()}
    raise RuntimeError("executed notebook did not report interpreter/import provenance")


def main() -> int:
    args = _parser().parse_args()
    artifacts = execute_notebook(args.output_dir)
    for name, path in artifacts.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
