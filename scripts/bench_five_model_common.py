"""Reproducible common-contract baseline for a fixed high-level evaluation.

Use the archived evidence runner so imports and revision checks are pinned.
This script prints one JSON record; timings are observations, not thresholds.
"""

from __future__ import annotations

import json
import statistics
import time
from unittest.mock import patch

import numpy as np
from astropy import units as u

from gragra.analytic.objects import uniform_sphere
from gragra.evaluation import EvaluationSetup
from gragra.geometry import Frame, Sphere
from gragra.targets import TargetPoints

REPEATS = 12
TARGET_COUNT = 128


def make_setup() -> EvaluationSetup:
    frame = Frame("benchmark-root")
    sphere = Sphere(np.array([0.5, -0.25, 0.75]) * u.m, 2.0 * u.m, frame)
    offsets = np.linspace(0.0, 1.0, TARGET_COUNT)
    positions = np.column_stack((3.0 + offsets, 0.25 + offsets, 0.5 + offsets))
    targets = TargetPoints(
        positions * u.m,
        tuple(f"target-{i}" for i in range(TARGET_COUNT)),
        frame,
    )
    return uniform_sphere(sphere, 2670.0 * u.kg / u.m**3).at(targets)


def elapsed_ms(callback) -> tuple[float, object]:
    start = time.perf_counter_ns()
    value = callback()
    return (time.perf_counter_ns() - start) / 1_000_000, value


def main() -> None:
    # Warm imports, metadata lookups, and any backend caches on a separate setup.
    make_setup().run()

    construction_ms = [elapsed_ms(make_setup)[0] for _ in range(REPEATS)]
    setup = make_setup()
    first_ms, first = elapsed_ms(setup.run)
    repeated_ms = [elapsed_ms(setup.run)[0] for _ in range(REPEATS)]

    call_count = 0
    original_validate = EvaluationSetup.validate

    def counted_validate(self):
        nonlocal call_count
        call_count += 1
        return original_validate(self)

    with patch.object(EvaluationSetup, "validate", counted_validate):
        checked = setup.run()

    for name in ("potential", "acceleration", "gradient"):
        np.testing.assert_array_equal(
            getattr(first.values, name).value,
            getattr(checked.values, name).value,
        )

    result = {
        "workload": "uniform_sphere_128_targets_all_observables",
        "repeats": REPEATS,
        "target_count": TARGET_COUNT,
        "construction_ms": construction_ms,
        "first_run_ms": first_ms,
        "repeated_run_ms": repeated_ms,
        "construction_median_ms": statistics.median(construction_ms),
        "repeated_median_ms": statistics.median(repeated_ms),
        "validate_calls_per_run": call_count,
        "result_equal_after_counted_run": True,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
