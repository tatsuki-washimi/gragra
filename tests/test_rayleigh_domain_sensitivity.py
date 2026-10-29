"""Tests for the bounded analytic Rayleigh box-sensitivity calculation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from gragra.observables.dipole import displacement_transfer_acceleration
from gragra.waves.rayleigh import rayleigh_displacement


def _load_script():
    path = (
        Path(__file__).parents[1]
        / "scripts"
        / "research_rayleigh_domain_sensitivity.py"
    )
    spec = importlib.util.spec_from_file_location(
        "research_rayleigh_domain_sensitivity", path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gll_box_blocks_preserve_element_quadrature_and_shared_nodes():
    domain = _load_script()
    box = domain.BoxSpec(length_m=500.0, depth_m=250.0, element_size_m=250.0)

    blocks = list(domain.iter_box_blocks(box, max_sources=125))
    coordinates = np.concatenate([block[0] for block in blocks])
    weights = np.concatenate([block[1] for block in blocks])

    assert box.element_shape == (2, 2, 1)
    assert box.element_count == 4
    assert coordinates.shape == (4 * 125, 3)
    assert weights.sum() == pytest.approx(500.0 * 500.0 * 250.0)
    assert np.unique(coordinates, axis=0).shape[0] < coordinates.shape[0]
    assert box.source_count == 4 * 125
    assert box.retained_shared_node_representations > 0


def test_nested_box_overlap_coordinates_and_weights_are_identical():
    domain = _load_script()
    inner = domain.BoxSpec(length_m=500.0, depth_m=250.0, element_size_m=250.0)
    outer = domain.BoxSpec(length_m=1000.0, depth_m=250.0, element_size_m=250.0)

    report = domain.compare_nested_overlap(inner, outer, max_sources=125)

    assert report["coordinate_exact_match"]
    assert report["quadrature_weight_exact_match"]
    assert report["overlap_element_count"] == inner.element_count
    assert report["overlap_source_count"] == inner.source_count


def test_blocked_box_integral_matches_one_shot():
    domain = _load_script()
    box = domain.BoxSpec(length_m=500.0, depth_m=250.0, element_size_m=250.0)
    targets = np.array([[3000.0, 3000.0, 150.0]])
    result = domain.integrate_box(
        box, targets=targets, source_block=125, kernel_chunk=64
    )

    blocks = list(domain.iter_box_blocks(box, max_sources=box.source_count))
    coordinates, quadrature = blocks[0]
    displacement = rayleigh_displacement(
        coordinates,
        (1.0, 0.0, 0.0),
        domain.WAVENUMBER_RAD_M,
        domain.VP_M_S,
        domain.VS_M_S,
        amplitude_m=domain.AMPLITUDE_M,
    )
    expected = displacement_transfer_acceleration(
        coordinates,
        domain.DENSITY_KG_M3 * quadrature,
        displacement,
        targets,
        chunk_size=64,
    )

    # Blocking changes floating-point addition order.  The absolute tolerance
    # covers roundoff in the symmetry-zero y component while the relative
    # tolerance controls the physically nonzero x/z components (~1e-16 here).
    np.testing.assert_allclose(result.values, expected, rtol=1e-14, atol=5e-30)
    assert result.source_count == box.source_count
    assert result.nonfinite_count == 0
    assert result.peak_rss_bytes > 0
    assert not result.values.flags.writeable


def test_saved_solver_summary_preserves_provenance_and_confound(tmp_path):
    domain = _load_script()
    a6 = tmp_path / "a6.json"
    a12 = tmp_path / "a12.json"
    row = {
        "window": "front_0_0.8s",
        "target": "0.25_lambda",
        "A_solver_finite_volume": [[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]],
    }
    a6.write_text(json.dumps({"source_hdf5": "/tmp/a6.h5", "results": [row]}))
    a12.write_text(json.dumps({"source_hdf5": "/tmp/a12.h5", "results": [row]}))

    summary = domain.summarize_saved_solver_results(a6, a12)

    assert summary["classification"] == "Observed saved-result reference"
    assert summary["not_a_solver_convergence_test"]
    assert summary["revision_and_configuration_confounds"]
    assert summary["a6"]["sha256"]
    assert summary["a12"]["sha256"]
    assert summary["a6"]["conditions_checked"]["target_coordinates_match"] is None


def test_targetwise_relative_l2_keeps_low_amplitude_target_visible():
    domain = _load_script()
    reference = np.array([[10.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
    calculated = np.array([[11.0, 0.0, 0.0], [0.02, 0.0, 0.0]])

    relative = domain.targetwise_relative_l2(calculated, reference)
    adjacent = domain.targetwise_difference_over_reference(
        calculated, reference, np.array([[20.0, 0.0, 0.0], [0.001, 0.0, 0.0]])
    )

    np.testing.assert_allclose(relative, [0.1, 1.0], rtol=1e-15, atol=0.0)
    np.testing.assert_allclose(adjacent, [0.05, 10.0], rtol=1e-15, atol=0.0)
