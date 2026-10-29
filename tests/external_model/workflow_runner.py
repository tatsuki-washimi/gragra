"""Run one external model workflow in a fresh interpreter."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from astropy import units as u

_SOURCE_ROOT = (Path(__file__).resolve().parents[2] / "src").resolve()
# A scikit-build editable finder can point at another worktree even when this
# subprocess receives the current worktree's src directory in PYTHONPATH.
sys.meta_path[:] = [
    finder
    for finder in sys.meta_path
    if "editable_skbc_gragra" not in type(finder).__module__
]
sys.path.insert(0, str(_SOURCE_ROOT))

import gragra  # noqa: E402

_SOURCE_FILE = Path(gragra.__file__).resolve()
if _SOURCE_ROOT not in _SOURCE_FILE.parents:
    raise RuntimeError(f"expected gragra under {_SOURCE_ROOT}, imported {_SOURCE_FILE}")

from fixture_models import ExternalPhasorModel, ExternalStaticModel  # noqa: E402

from gragra import TargetPoints  # noqa: E402
from gragra.geometry import Frame  # noqa: E402
from gragra.results import compare  # noqa: E402


def _forbid_run() -> None:
    raise AssertionError("result operation called evaluation.run")


def main(kind: str) -> None:
    models = {"static": ExternalStaticModel, "phasor": ExternalPhasorModel}
    model_type = models[kind]
    frame = Frame("external-world")
    targets = TargetPoints(np.zeros((1, 3)) * u.m, ("single",), frame)
    reference_evaluation = model_type().at(targets)
    candidate_evaluation = model_type(offset=1.0).at(targets)
    reference = reference_evaluation.run()
    candidate = candidate_evaluation.run()

    assert reference.evaluation is reference_evaluation
    assert candidate.evaluation is candidate_evaluation
    assert reference.context is reference_evaluation.context
    assert reference.context.targets is targets
    assert reference.context.frame is frame
    assert reference.context.interaction.name == "newton"
    assert dict(reference.context.interaction.parameters) == {}
    assert reference.context.binding("potential") is reference.bindings["potential"]
    assert reference.context.binding("gradient") is reference.bindings["gradient"]
    assert reference.parameters["offset"][0] == 0.0
    assert reference.formulas.empty and reference.references.empty
    assert reference.g_applications == 1

    if kind == "static":
        assert reference.context.frequencies_hz is None
        assert reference.context.reference_time_gps is None
        assert reference.potential.shape == (1,)
        assert reference.acceleration.shape == (1, 3)
        assert reference.gradient.shape == (1, 3, 3)
        assert reference.potential.unit.is_equivalent(u.m**2 / u.s**2)
    else:
        assert reference.context.frequencies_hz == (0.25,)
        assert reference.context.reference_time_gps == (1_000_000_000, 123)
        assert reference.potential.shape == (1, 1, 1)
        assert reference.acceleration.shape == (1, 3, 1)
        assert reference.gradient.shape == (1, 9, 1)
        assert tuple(reference.potential.rows) == ("single",)
        assert tuple(reference.potential.cols) == ("potential",)
        assert tuple(reference.acceleration.cols) == ("x", "y", "z")
        assert tuple(reference.gradient.cols) == (
            "xx",
            "xy",
            "xz",
            "yx",
            "yy",
            "yz",
            "zx",
            "zy",
            "zz",
        )
        assert reference.potential.meta[0, 0].unit.is_equivalent(u.m**2 / u.s**2)
        np.testing.assert_array_equal(
            reference.potential.frequencies.to_value(u.Hz), [0.25]
        )

    reference_evaluation.run = _forbid_run
    candidate_evaluation.run = _forbid_run
    selected_reference = reference.select("gradient", "potential")
    selected_candidate = candidate.select("gradient", "potential")
    assert selected_reference.observables == ("gradient", "potential")
    assert selected_reference.context is reference.context
    assert selected_reference.evaluation is reference.evaluation
    assert selected_reference.record is reference.record
    assert selected_reference.values.gradient is reference.values.gradient
    assert selected_reference.values.potential is reference.values.potential
    assert selected_reference.bindings["gradient"] is reference.bindings["gradient"]
    assert selected_reference.bindings["potential"] is reference.bindings["potential"]

    comparison = compare(reference=selected_reference, candidate=selected_candidate)
    assert comparison.reference is selected_reference
    assert comparison.candidate is selected_candidate
    if kind == "static":
        difference = float(
            comparison.differences.potential.to_value(u.m**2 / u.s**2)[0]
        )
    else:
        value = np.asarray(comparison.differences.potential)[0, 0, 0]
        assert value.imag == 0.0
        difference = float(value.real)
    assert difference == 1.0
    print(
        json.dumps(
            {
                "kind": kind,
                "source": str(_SOURCE_FILE),
                "target_count": len(targets),
                "frequency_count": len(reference.context.frequencies_hz or ()),
                "runs_per_evaluation": [
                    reference_evaluation.run_count,
                    candidate_evaluation.run_count,
                ],
                "selected_observables": list(selected_reference.observables),
                "difference": difference,
            }
        )
    )


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"static", "phasor"}:
        raise SystemExit("usage: workflow_runner.py {static,phasor}")
    main(sys.argv[1])
