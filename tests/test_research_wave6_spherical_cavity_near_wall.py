"""Small regression checks for the off-axis near-wall research reference."""

import numpy as np

import scripts.research_wave6_spherical_cavity_near_wall as study
from scripts.research_wave6_spherical_cavity_near_wall import (
    _volume_at_order,
    _wall_at_order,
)


def _case():
    radial_direction = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    return {
        "target": 0.99 * radial_direction,
        "radius": 1.0,
        "density": 2700.0,
        "wavenumber": 0.8,
        "displacement": (1.0 + 0.4j) * 1.0e-6,
        "direction": np.array([0.0, 0.0, 1.0]),
        "polarization": np.array([0.0, 0.0, 1.0]),
        "wave_type": "P",
    }


def test_split_wall_refines_without_nonfinite_values():
    case = _case()
    wall_case = {key: value for key, value in case.items() if key != "wave_type"}
    low = _wall_at_order(**wall_case, order=16)
    high = _wall_at_order(**wall_case, order=32)

    for name in ("potential", "acceleration", "gradient"):
        assert np.isfinite(low[name]).all()
        assert np.isfinite(high[name]).all()
        change = np.linalg.norm(high[name] - low[name])
        scale = max(np.linalg.norm(high[name]), np.finfo(float).tiny)
        assert change / scale < 1.0e-4


def test_split_volume_returns_all_observables_for_interior_target():
    case = _case()
    values = _volume_at_order(**case, order=16)

    assert set(values) == {"potential", "acceleration", "gradient"}
    assert values["potential"].shape == (1,)
    assert values["acceleration"].shape == (1, 3)
    assert values["gradient"].shape == (1, 3, 3)
    for value in values.values():
        assert np.isfinite(value).all()


def test_run_pilot_passes_condition_and_order_parameters(monkeypatch):
    observed = []

    def compare_case(target, **kwargs):
        observed.append((target.copy(), kwargs))
        return {
            "qscale_normalized_reference_change": {},
            "qscale_normalized_production_error": {},
        }

    monkeypatch.setattr(study, "compare_case", compare_case)
    result = study.run_pilot(
        d_over_a=1.0e-3,
        wavenumber=5.0,
        low_order=128,
        high_order=256,
    )

    assert result["d_over_a"] == 1.0e-3
    assert result["wavenumber_rad_m"] == 5.0
    assert result["reference_orders"] == [128, 256]
    assert len(observed) == 8
    assert all(call[1]["low_order"] == 128 for call in observed)
    assert all(call[1]["high_order"] == 256 for call in observed)
