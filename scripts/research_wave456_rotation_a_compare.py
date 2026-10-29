#!/usr/bin/env python3
"""Compare stored R motion-source results with parent-accepted A references."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import resource
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.meta_path[:] = [
    finder
    for finder in sys.meta_path
    if type(finder).__name__ != "ScikitBuildRedirectingFinder"
]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from scripts.research_wave456_rotation import (  # noqa: E402
    PASSING_TARGETS_M,
    ROTATION_TARGETS_M,
    _comparison_by_target,
    _complex_pairs,
    _phasors,
    _target_norm,
)

_ACCEPTED_ANALYTIC_SHA256 = (
    "bda1d6075ad042511d51624d8bca8e9b244d7e7063f401c12f2c4eba4cee11be"
)
_ACCEPTED_WRAPPER_SHA256 = (
    "7ebe9850fa6fa242fcc1be3f3635c6436c3d60c8bd3878a75423ba6c7701af47"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_accepted_a(a_root: Path) -> ModuleType:
    """Load the accepted A wrapper without copying it into the R worktree."""
    analytic_path = a_root / "src/gragra/analytic/wave456_reference.py"
    wrapper_path = a_root / "src/gragra/observables/wave456_reference.py"
    if _sha256(analytic_path) != _ACCEPTED_ANALYTIC_SHA256:
        raise RuntimeError("A analytic source SHA-256 does not match parent acceptance")
    if _sha256(wrapper_path) != _ACCEPTED_WRAPPER_SHA256:
        raise RuntimeError(
            "A observable wrapper SHA-256 does not match parent acceptance"
        )
    analytic_spec = importlib.util.spec_from_file_location(
        "gragra.analytic.wave456_reference", analytic_path
    )
    assert analytic_spec is not None and analytic_spec.loader is not None
    analytic = importlib.util.module_from_spec(analytic_spec)
    sys.modules[analytic_spec.name] = analytic
    analytic_spec.loader.exec_module(analytic)
    wrapper_spec = importlib.util.spec_from_file_location(
        "r_accepted_a_wrapper", wrapper_path
    )
    assert wrapper_spec is not None and wrapper_spec.loader is not None
    wrapper = importlib.util.module_from_spec(wrapper_spec)
    wrapper_spec.loader.exec_module(wrapper)
    return wrapper


def _from_pairs(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.shape == (2,) and array.dtype.kind in "fiu":
        return np.complex128(array[0] + 1j * array[1])
    return np.asarray([_from_pairs(item) for item in value])


def _observable_mapping(value: Any) -> dict[str, np.ndarray]:
    return {
        "potential": np.asarray(value.potential),
        "acceleration": np.asarray(value.acceleration),
        "gradient": np.asarray(value.gradient),
    }


def _phasor_comparison(
    production: dict[str, Any],
    reference: dict[str, np.ndarray],
    harmonic: str,
    h2_reference: dict[str, np.ndarray],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for observable, reference_value in reference.items():
        production_value = _from_pairs(production[observable][harmonic])
        comparison = _comparison_by_target(
            production_value[None, ...], reference_value[None, ...], observable
        )
        comparison.pop("relative_l2_per_target")
        h2_scale = _target_norm(h2_reference[observable][None, ...], observable)[0]
        resolution_floor = np.finfo(np.float64).eps * h2_scale
        absolute = _target_norm(
            production_value[None, ...] - reference_value[None, ...], observable
        )[0]
        own_scale = _target_norm(reference_value[None, ...], observable)[0]
        comparison["absolute_norm_per_target"] = absolute.tolist()
        comparison["normalized_by_h2_per_target"] = [
            float(value / scale) if scale > 0.0 else None
            for value, scale in zip(absolute, h2_scale, strict=True)
        ]
        comparison["h2_scale_per_target"] = h2_scale.tolist()
        comparison["resolution_floor_per_target"] = resolution_floor.tolist()
        comparison["own_harmonic_relative_per_target"] = [
            float(value / scale) if scale > floor else None
            for value, scale, floor in zip(
                absolute, own_scale, resolution_floor, strict=True
            )
        ]
        result[observable] = comparison
    return result


def _temporal_sampling_curve(
    by_key: dict[tuple[tuple[int, int, int], int], dict[str, Any]],
) -> dict[str, Any]:
    """Compare saved finite phasors at fixed quadrature against N=1024."""
    quadrature = (16, 64, 16)
    reference = by_key[(quadrature, 1024)]["finite_phasors"]
    return {
        str(samples): {
            harmonic: _phasor_comparison(
                by_key[(quadrature, samples)]["finite_phasors"],
                {name: _from_pairs(reference[name][harmonic]) for name in reference},
                harmonic,
                {name: _from_pairs(reference[name]["2"]) for name in reference},
            )
            for harmonic in ("dc", *(str(index) for index in range(1, 13)))
        }
        for samples in (256, 512)
    }


def _a_rotor_phasors(
    reference: ModuleType, angles: np.ndarray, order: tuple[int, int, int]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    value = reference.opposed_rotor_observables(
        angles,
        ROTATION_TARGETS_M,
        radial_order=order[0],
        azimuthal_order=order[1],
        axial_order=order[2],
    )
    mapped = _observable_mapping(value)
    return {name: _phasors(series, angles) for name, series in mapped.items()}, mapped


def _curve_entry(
    candidate: dict[str, np.ndarray], reference: dict[str, np.ndarray]
) -> dict[str, Any]:
    return {
        observable: _comparison_by_target(
            candidate[observable], reference[observable], observable
        )
        for observable in candidate
    }


def _passing_reference(
    reference: ModuleType, times_s: np.ndarray, *, order: int = 24
) -> dict[str, np.ndarray]:
    """Evaluate the A cuboid at the fixed passing-body targets, never rotor targets."""
    return _observable_mapping(
        reference.passing_cuboid_observables(times_s, PASSING_TARGETS_M, order=order)
    )


def _maximum_relative_difference(curve: dict[str, Any]) -> float:
    return max(
        value
        for observable in curve.values()
        for value in observable["relative_l2_per_target"]
    )


def _passing_order_convergence(
    reference: ModuleType, times_s: np.ndarray, order24: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Check every passing frame against independent Gauss order refinements."""
    order10 = _passing_reference(reference, times_s, order=10)
    order12 = _passing_reference(reference, times_s, order=12)
    ten_to_twelve = _curve_entry(order10, order12)
    twelve_to_twentyfour = _curve_entry(order12, order24)
    maximum = max(
        _maximum_relative_difference(ten_to_twelve),
        _maximum_relative_difference(twelve_to_twentyfour),
    )
    return {
        "frames": int(len(times_s)),
        "orders": [10, 12, 24],
        "order10_vs12_by_target": ten_to_twelve,
        "order12_vs24_by_target": twelve_to_twentyfour,
        "maximum_relative_l2": maximum,
        "threshold": 1.0e-11,
        "status": "CONVERGED" if maximum <= 1.0e-11 else "ORACLE_UNRESOLVED",
    }


def run(results_path: Path, output_path: Path, a_root: Path) -> Path:
    """Write independent cylinder, multipole, and passing-body comparisons."""
    if output_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing comparison: {output_path}"
        )
    stored = json.loads(results_path.read_text())
    if (
        stored["source_identity"]["script_sha256"]
        != hashlib.sha256(
            (_ROOT / "scripts/research_wave456_rotation.py").read_bytes()
        ).hexdigest()
    ):
        raise RuntimeError("stored result does not match this R script SHA-256")
    reference = _load_accepted_a(a_root)
    cases = stored["rotation"]["cases"]
    by_key = {
        (tuple(case["quadrature"]), case["samples_per_rotation"]): case
        for case in cases
    }
    a_cases: dict[tuple[int, int], tuple[dict[str, Any], dict[str, np.ndarray]]] = {}
    for samples in (256, 512, 1024):
        angles = 2.0 * np.pi * np.arange(samples, dtype=np.float64) / samples
        a_cases[(24, samples)] = _a_rotor_phasors(reference, angles, (24, 96, 24))

    production_curves: dict[str, Any] = {}
    for quadrature in ((4, 16, 4), (8, 32, 8), (16, 64, 16)):
        for samples in (256, 512, 1024):
            stored_case = by_key[(quadrature, samples)]
            a_phasors, _ = a_cases[(24, samples)]
            production_curves[f"q{quadrature}_n{samples}"] = {
                harmonic: _phasor_comparison(
                    stored_case["finite_phasors"],
                    {
                        name: _from_pairs(a_phasors[name][harmonic])
                        for name in a_phasors
                    },
                    harmonic,
                    {name: _from_pairs(a_phasors[name]["2"]) for name in a_phasors},
                )
                for harmonic in ("dc", *(str(index) for index in range(1, 13)))
            }

    order_curves: dict[str, Any] = {}
    angular_curves: dict[str, Any] = {}
    angles_256 = 2.0 * np.pi * np.arange(256, dtype=np.float64) / 256
    order_values: dict[int, dict[str, np.ndarray]] = {}
    for order in (10, 12, 16, 24):
        _, order_values[order] = _a_rotor_phasors(
            reference, angles_256, (order, 96, order)
        )
    for order in (10, 12, 16):
        order_curves[str(order)] = _curve_entry(order_values[order], order_values[24])
    angular_values: dict[int, dict[str, np.ndarray]] = {}
    for azimuthal in (16, 32, 64, 128):
        _, angular_values[azimuthal] = _a_rotor_phasors(
            reference, angles_256, (24, azimuthal, 24)
        )
    for azimuthal in (16, 32, 64):
        angular_curves[str(azimuthal)] = _curve_entry(
            angular_values[azimuthal], angular_values[128]
        )

    leading = _observable_mapping(
        reference.opposed_rotor_leading_phasor(ROTATION_TARGETS_M)
    )
    finest = by_key[((16, 64, 16), 1024)]["finite_phasors"]
    leading_comparison = _phasor_comparison(finest, leading, "2", leading)

    raw_path = Path(stored["passing"]["raw_series_path"])
    raw = np.load(raw_path)
    passing = _passing_reference(reference, raw["time_s"])
    passing_comparison = {
        observable: _comparison_by_target(
            raw[f"finite_{observable}"], passing[observable], observable
        )
        for observable in ("potential", "acceleration", "gradient")
    }
    passing_convergence = _passing_order_convergence(reference, raw["time_s"], passing)
    result = {
        "a_dependency": {
            "accepted_analytic_sha256": _ACCEPTED_ANALYTIC_SHA256,
            "accepted_wrapper_sha256": _ACCEPTED_WRAPPER_SHA256,
            "acceptance_record": (
                "/home/washimi/work/gragra-research/2026-09-12-wave4-6/"
                "verification/A-parent-acceptance.json"
            ),
            "root": str(a_root),
        },
        "r_result": str(results_path),
        "production_artifact_scope": (
            "current3 production artifact is production-only; this companion "
            "report supplies the parent-accepted A validation and does not "
            "modify the stored production calculation."
        ),
        "r_comparison_script_sha256": _sha256(Path(__file__).resolve()),
        "generation": {
            "argv": sys.argv,
            "utc": datetime.now(UTC).isoformat(),
            "peak_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
            "input_result_sha256": _sha256(results_path),
            "input_passing_raw_sha256": _sha256(raw_path),
        },
        "stored_phase_convention": stored["rotation"]["phase_convention"],
        "production_discretization_curves_by_target": production_curves,
        "finite_phasor_time_sampling_at_fixed_quadrature": _temporal_sampling_curve(
            by_key
        ),
        "raw_a_phasors": {
            f"n{samples}": a_cases[(24, samples)][0] for samples in (256, 512, 1024)
        },
        "independent_cylinder_order_curve_by_target": order_curves,
        "independent_cylinder_angular_curve_by_target": angular_curves,
        "leading_nonzero_multipole": {
            "harmonic": 2,
            "phase_convention": "Re[q_hat exp(-i 2 theta)]",
            "phasor": {name: _complex_pairs(value) for name, value in leading.items()},
            "finite_h2_vs_multipole_by_target": leading_comparison,
            "scope": (
                "far second-moment approximation; not expected to equal finite "
                "cylinder values"
            ),
        },
        "passing_cuboid_independent_order24_by_target": passing_comparison,
        "passing_cuboid_independent_order_convergence": passing_convergence,
    }
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--a-root",
        type=Path,
        default=Path(
            "/home/washimi/.config/superpowers/worktrees/gragra/wave456-20260912-A"
        ),
    )
    args = parser.parse_args()
    print(run(args.results, args.output, args.a_root))


if __name__ == "__main__":
    main()
