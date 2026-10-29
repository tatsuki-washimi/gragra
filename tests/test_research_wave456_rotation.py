"""Behavioural checks for the fixed Wave 4--6 motion-source study."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.research_wave456_rotation import (
    _comparison_by_target,
    _passing_case,
    _rotation_transform,
    cylinder_cloud,
    finite_difference_target_shift,
    rotating_target_shift,
    rotation_z,
)
from scripts.research_wave456_rotation_a_compare import (
    _passing_order_convergence,
    _passing_reference,
    _phasor_comparison,
)


def test_research_script_imports_this_worktree_source():
    import gragra

    repository_root = Path(__file__).resolve().parents[1]
    assert Path(gragra.__file__).resolve().is_relative_to(repository_root / "src")


def test_opposed_cylinder_cloud_has_fixed_total_mass_and_com_limit():
    cloud = cylinder_cloud((4, 16, 4), radius_m=0.1, length_m=0.2, density_kg_m3=7800.0)
    expected_mass = 2.0 * np.pi * 0.1**2 * 0.2 * 7800.0
    assert cloud.masses_kg.dtype == np.float64
    assert np.isclose(cloud.masses_kg.sum(), expected_mass, rtol=1e-14)
    assert np.allclose(
        np.average(cloud.positions_m, axis=0, weights=cloud.masses_kg),
        [0.0, 0.0, 0.0],
        atol=1e-15,
    )
    assert np.allclose(rotation_z(np.array([0.0]))[0], np.eye(3))


def test_target_shift_gradient_prediction_has_correct_sign_and_converges():
    cloud = cylinder_cloud((4, 16, 4), radius_m=0.1, length_m=0.2, density_kg_m3=7800.0)
    result = finite_difference_target_shift(
        cloud, np.array([[2.0, 0.3, 0.2]]), axis=0, step_m=1.0e-3
    )
    assert result["relative_error"] < 2.0e-6
    assert np.allclose(
        result["central_difference"], result["gradient_prediction"], rtol=2e-6
    )


def test_per_target_metrics_and_rotating_gradient_prediction_preserve_targets():
    cloud = cylinder_cloud((4, 16, 4), radius_m=0.1, length_m=0.2, density_kg_m3=7800.0)
    targets = np.array([[2.0, 0.3, 0.2], [4.0, 1.0, 0.5]])
    shift = rotating_target_shift(
        cloud, _rotation_transform(256), targets, axis=1, step_m=1.0e-3
    )
    assert shift["central_difference"].shape == (256, 2, 3)
    assert shift["gradient_prediction"].shape == (256, 2, 3)
    metrics = _comparison_by_target(
        shift["central_difference"], shift["gradient_prediction"], "acceleration"
    )
    assert len(metrics["max_abs_per_target"]) == 2
    assert max(metrics["relative_l2_per_target"]) < 2.0e-6


def test_gradient_comparison_uses_per_target_frobenius_norm():
    reference = np.ones((2, 2, 3, 3), dtype=np.float64)
    candidate = reference.copy()
    candidate[:, 1, 0, 0] += 3.0
    metrics = _comparison_by_target(candidate, reference, "gradient")
    assert metrics["max_abs_per_target"] == [0.0, 3.0]


def test_passing_case_retains_three_observables_as_raw_series():
    summary, raw = _passing_case()
    assert summary["samples"] == 401
    assert raw["time_s"].shape == (401,)
    assert raw["finite_potential"].shape == (401, 3)
    assert raw["finite_acceleration"].shape == (401, 3, 3)
    assert raw["finite_gradient"].shape == (401, 3, 3, 3)
    assert raw["com_potential"].shape == (401, 3)
    assert raw["com_acceleration"].shape == (401, 3, 3)
    assert raw["com_gradient"].shape == (401, 3, 3, 3)


def test_independent_passing_reference_receives_passing_targets_not_rotor_targets():
    captured: dict[str, np.ndarray] = {}

    class FakeReference:
        def passing_cuboid_observables(self, times_s, targets_m, *, order):  # noqa: ARG002
            captured["targets_m"] = np.asarray(targets_m)
            samples, targets = len(times_s), len(targets_m)
            return SimpleNamespace(
                potential=np.zeros((samples, targets)),
                acceleration=np.zeros((samples, targets, 3)),
                gradient=np.zeros((samples, targets, 3, 3)),
            )

    _passing_reference(FakeReference(), np.array([-2.0, 2.0]))
    assert np.array_equal(
        captured["targets_m"], [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [3.0, -2.0, 1.0]]
    )


def test_independent_passing_order_convergence_covers_every_frame_and_observable():
    orders: list[int] = []

    class FakeReference:
        def passing_cuboid_observables(self, times_s, targets_m, *, order):
            orders.append(order)
            samples, targets = len(times_s), len(targets_m)
            return SimpleNamespace(
                potential=np.zeros((samples, targets)),
                acceleration=np.zeros((samples, targets, 3)),
                gradient=np.zeros((samples, targets, 3, 3)),
            )

    order24 = _passing_reference(FakeReference(), np.array([-2.0, 0.0, 2.0]))
    convergence = _passing_order_convergence(
        FakeReference(), np.array([-2.0, 0.0, 2.0]), order24
    )
    assert orders == [24, 10, 12]
    assert convergence["frames"] == 3
    assert convergence["status"] == "CONVERGED"
    assert set(convergence["order10_vs12_by_target"]) == {
        "potential",
        "acceleration",
        "gradient",
    }


def test_opposed_rotor_has_pi_periodicity_but_not_odd_harmonic_only_control():
    cloud = cylinder_cloud((4, 16, 4), radius_m=0.1, length_m=0.2, density_kg_m3=7800.0)
    theta = 2.0 * np.pi * np.arange(64) / 64
    rotations = rotation_z(theta)
    rotated = cloud.positions_m @ rotations[0].T
    rotated_pi = cloud.positions_m @ rotations[32].T
    assert np.allclose(np.sort(rotated, axis=0), np.sort(rotated_pi, axis=0))
    # A lone off-centre cylinder is the deliberate negative control: its
    # half-period samples do not represent the same source arrangement.
    one_body = cloud.positions_m[cloud.positions_m[:, 0] > 0.0]
    assert not np.allclose(
        np.sort(one_body @ rotations[0].T, axis=0),
        np.sort(one_body @ rotations[32].T, axis=0),
    )


def test_phasor_comparison_keeps_per_target_h2_scales_for_tiny_harmonics():
    def pairs(value):
        array = np.asarray(value)
        if array.ndim == 0:
            return [array.real.item(), array.imag.item()]
        return [pairs(item) for item in array]

    reference = {
        "potential": np.array([0.0j, 4.0j]),
        "acceleration": np.array([[0.0j, 0.0j, 0.0j], [4.0j, 0.0j, 0.0j]]),
        "gradient": np.zeros((2, 3, 3), dtype=np.complex128),
    }
    production = {name: {"1": pairs(values)} for name, values in reference.items()}
    h2 = {
        "potential": np.array([2.0, 8.0]),
        "acceleration": np.array([[2.0, 0.0, 0.0], [8.0, 0.0, 0.0]]),
        "gradient": np.array([np.eye(3) * 2.0, np.eye(3) * 8.0]),
    }
    result = _phasor_comparison(production, reference, "1", h2)
    for comparison in result.values():
        assert set(comparison) >= {
            "absolute_norm_per_target",
            "max_abs_per_target",
            "normalized_by_h2_per_target",
            "h2_scale_per_target",
            "resolution_floor_per_target",
            "own_harmonic_relative_per_target",
        }
        assert len(comparison["h2_scale_per_target"]) == 2
        assert comparison["own_harmonic_relative_per_target"][0] is None
