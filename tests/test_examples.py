"""Smoke-run tests for examples/ scripts.

Verifies that:
- All expected example files exist (drift guard).
- Each example runs to completion via ``runpy.run_path`` with
  ``run_name="__main__"`` so that ``if __name__ == "__main__": main()``
  guards are triggered.
- Execution happens inside ``tmp_path``; no files are written to cwd.
- The meshio-dependent example is skipped when meshio is not installed.
"""

from __future__ import annotations

import importlib.util
import runpy
from pathlib import Path

import pytest

EXAMPLES_DIR: Path = Path(__file__).parent.parent / "examples"

EXPECTED_EXAMPLES: list[str] = [
    "01_point_mass.py",
    "02_weighted_point_source.py",
    "03_geometry_primitives.py",
    "04_analytic_benchmark.py",
    "05_gravity_gradient_tensor.py",
    "06_continuous_sources.py",
    "07_coherent_response.py",
    "08_meshio_surface_adapter.py",
    "09_external_field_density.py",
    "10_halfspace_rayleigh.py",
    "12_cross_statistics_witness.py",
    "13_sca_lower_limit_gate.py",
    "gravity_result.py",
]

# -----------------------------------------------------------------------
# Existence guard (RED until all examples are created)
# -----------------------------------------------------------------------


@pytest.mark.parametrize("filename", EXPECTED_EXAMPLES)
def test_expected_examples_exist(filename: str) -> None:
    """Assert that every expected example file is present.

    This test is intentionally RED until each example script is written,
    preventing number-skips or silent drift in the examples list.
    """
    path = EXAMPLES_DIR / filename
    assert path.is_file(), (
        f"Expected example file not found: {path}\n"
        "Create the file or update EXPECTED_EXAMPLES if the name changed."
    )


# -----------------------------------------------------------------------
# Smoke-run parametrization
# -----------------------------------------------------------------------


def _should_skip_meshio(filename: str) -> tuple[bool, str]:
    """Return (skip, reason) for meshio-dependent examples."""
    if (
        filename == "08_meshio_surface_adapter.py"
        and importlib.util.find_spec("meshio") is None
    ):
        return True, "meshio is not installed (pip install 'gragra[mesh]')"
    return False, ""


@pytest.mark.parametrize("filename", EXPECTED_EXAMPLES)
def test_example_smoke_run(
    filename: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Smoke-run each example inside tmp_path with no leftover files.

    Implementation notes
    --------------------
    - ``monkeypatch.chdir(tmp_path)`` redirects cwd so that any accidental
      file writes go into the temporary directory instead of the repo root.
    - ``runpy.run_path(..., run_name="__main__")`` is required because
      examples use the ``if __name__ == "__main__": main()`` guard.
      Omitting ``run_name`` would leave ``main()`` uncalled.
    - After execution the set of files in ``tmp_path`` must equal the set
      before execution (empty), confirming that the example produces no
      output artefacts.
    """
    skip, reason = _should_skip_meshio(filename)
    if skip:
        pytest.skip(reason)

    path = EXAMPLES_DIR / filename
    if not path.is_file():
        pytest.skip(f"Example file not yet created: {filename}")

    monkeypatch.chdir(tmp_path)

    files_before: set[Path] = set(tmp_path.iterdir())

    runpy.run_path(str(path), run_name="__main__")

    files_after: set[Path] = set(tmp_path.iterdir())
    new_files = files_after - files_before
    assert not new_files, (
        f"Example '{filename}' left artefacts in tmp_path: "
        + ", ".join(str(f.name) for f in sorted(new_files))
    )
