# ruff: noqa: E402, I001
"""Stage-5 (5-D) tests: spatial decomposition of the Rayleigh gravity window gap."""

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

from scripts.research_rayleigh_gravity_window_diagnostics import (
    REGION_NAMES,
    region_gravity_window_gap,
    region_masks,
    run,
)
from test_research_wave5_rayleigh_crosspaths import canonical_rayleigh_hdf5  # noqa: F401

_FIXTURE_DOMAIN_BOUNDS_M = ((-180.0, 540.0), (-140.0, 140.0), (-300.0, -20.0))


def test_region_masks_are_disjoint_exhaustive_and_classify_known_points():
    positions = np.array(
        [
            [1.0, 500.0, -1500.0],  # near x=0 lateral edge
            [3000.0, 3000.0, -1.0],  # near free surface (z=0), far from other edges
            [3000.0, 3000.0, -2999.0],  # near bottom (z=-3000)
            [3000.0, 3000.0, -1500.0],  # deep interior, far from all edges
            [5999.0, 3000.0, -1500.0],  # near x=6000 lateral edge
        ]
    )
    masks = region_masks(
        positions,
        domain_bounds_m=((0.0, 6000.0), (0.0, 6000.0), (-3000.0, 0.0)),
        boundary_distance_m=500.0,
    )
    assert set(masks) == set(REGION_NAMES)
    stacked = np.stack([masks[name] for name in REGION_NAMES])
    np.testing.assert_array_equal(stacked.sum(axis=0), np.ones(5, dtype=np.int64))
    assert masks["lateral_boundary_near"][0]
    assert masks["surface_near"][1]
    assert masks["bottom_boundary_near"][2]
    assert masks["interior"][3]
    assert masks["lateral_boundary_near"][4]


def test_region_masks_reject_non_positive_boundary_distance():
    positions = np.array([[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="boundary_distance_m"):
        region_masks(
            positions,
            domain_bounds_m=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
            boundary_distance_m=0.0,
        )


def test_region_gravity_window_gap_region_sum_reproduces_total(
    canonical_rayleigh_hdf5,
):
    path, _, _, _ = canonical_rayleigh_hdf5
    result = region_gravity_window_gap(
        path,
        boundary_distance_wavelengths=0.3,
        domain_bounds_m=_FIXTURE_DOMAIN_BOUNDS_M,
        source_block_elements=1,
    )
    assert set(result["region_gap_front_minus_full"]) == set(REGION_NAMES)
    assert sum(result["region_source_counts"].values()) == 250
    for observable, stats in result["consistency_check_region_sum_vs_total"].items():
        scale = max(stats["max_total_norm"], 1e-300)
        assert stats["max_absolute_difference"] / scale < 1e-9, observable


def test_run_writes_complete_gravity_diagnostic_json(canonical_rayleigh_hdf5, tmp_path):
    path, _, _, _ = canonical_rayleigh_hdf5
    output = tmp_path / "outside" / "gravity-diagnostics.json"
    result_path = run(path, output, source_block_elements=1)
    assert result_path == output
    payload = json.loads(output.read_text())
    assert payload["status"] == "complete"
    assert set(payload["region_gap_front_minus_full"]) == set(REGION_NAMES)
    with pytest.raises(FileExistsError):
        run(path, output, source_block_elements=1)
