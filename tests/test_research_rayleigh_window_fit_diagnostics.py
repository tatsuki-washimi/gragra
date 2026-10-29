# ruff: noqa: E402, I001
"""Stage-4 diagnostic tests for the saved-Rayleigh window-fit gap."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

WORKTREE = Path(__file__).resolve().parents[1]
sys.meta_path[:] = [
    item
    for item in sys.meta_path
    if type(item).__name__ != "ScikitBuildRedirectingFinder"
]
for entry in (WORKTREE / "src", WORKTREE):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from scripts.research_rayleigh_window_fit_diagnostics import (
    design_matrix_condition_numbers,
    run,
    window_fit_residual_rms,
)
from test_research_wave5_rayleigh_crosspaths import canonical_rayleigh_hdf5  # noqa: F401


def test_design_matrix_condition_numbers_are_finite_and_reproduce_from_time_contract():
    result = design_matrix_condition_numbers()
    assert set(result) == {"front", "full"}
    for window_name, value in result.items():
        assert np.isfinite(value)
        assert value > 0.0, window_name


def test_window_fit_residual_rms_is_near_zero_for_solver_and_analytic_on_synthetic_fixture(
    canonical_rayleigh_hdf5,
):
    path, _, _, _ = canonical_rayleigh_hdf5
    # The synthetic fixture's saved "solver" displacement is built directly
    # from rayleigh_displacement, so both diagnostics should recover
    # near-zero residual (floating-point only) and agree closely with
    # each other on this fixture.
    solver = window_fit_residual_rms(
        path, source="solver", source_block_elements=1, memory_budget_bytes=2 * 1024**3
    )
    analytic = window_fit_residual_rms(
        path,
        source="analytic",
        source_block_elements=1,
        memory_budget_bytes=2 * 1024**3,
    )
    for window_name in ("front", "full"):
        assert solver[window_name]["residual_to_signal_ratio"] < 1e-9
        assert analytic[window_name]["residual_to_signal_ratio"] < 1e-9
        assert solver[window_name]["sample_count"] == analytic[window_name]["sample_count"]
        assert solver[window_name]["source_count"] == 250


def test_window_fit_residual_rms_rejects_unknown_source(canonical_rayleigh_hdf5):
    path, _, _, _ = canonical_rayleigh_hdf5
    with pytest.raises(ValueError, match="source"):
        window_fit_residual_rms(path, source="bogus")


def test_run_writes_complete_diagnostic_json_and_refuses_overwrite(
    canonical_rayleigh_hdf5, tmp_path
):
    path, _, _, _ = canonical_rayleigh_hdf5
    output = tmp_path / "outside" / "diagnostics.json"
    result_path = run(path, output, source_block_elements=1)
    assert result_path == output
    payload = json.loads(output.read_text())
    assert payload["status"] == "complete"
    assert set(payload["diagnostic_1_design_matrix_condition_number"]) == {
        "front",
        "full",
    }
    assert "front" in payload["diagnostic_2_solver_fit_residual"]
    assert "front" in payload["diagnostic_3_analytic_fit_residual"]
    with pytest.raises(FileExistsError):
        run(path, output, source_block_elements=1)
