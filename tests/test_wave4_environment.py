from __future__ import annotations

from pathlib import Path

import pytest

import gragra
from scripts.research_wave4_environment import validate_free_surface_result


def test_imports_this_worktree_source():
    expected = Path(__file__).resolve().parents[1] / "src/gragra/__init__.py"
    assert Path(gragra.__file__).resolve() == expected


def test_free_surface_consumer_requires_signed_nonzero_mass_and_zero_controls():
    result = {
        "column_bounds_m": [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
        "densities_kg_m3": [1000.0, 1.2],
        "alphas_m": [-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02],
        "combination_policy": "compare_only_do_not_add",
        "mass_change_zero_assumed": False,
        "signed_mass_change_kg": {
            "-0.02": -1.0,
            "-0.01": -0.5,
            "-0.005": -0.25,
            "0.0": 0.0,
            "0.005": 0.25,
            "0.01": 0.5,
            "0.02": 1.0,
        },
        "controls": {
            "zero_displacement_exact_zero": True,
            "zero_density_jump_exact_zero": True,
            "sign_reversal_linear_sheet": True,
        },
        "responses": {
            str(value): {} for value in (-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02)
        },
    }

    validate_free_surface_result(result)
    broken = {**result, "mass_change_zero_assumed": True}
    with pytest.raises(ValueError, match="mass"):
        validate_free_surface_result(broken)
