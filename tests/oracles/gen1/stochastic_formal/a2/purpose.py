"""F_common_v2 four-field purpose records before any reference evaluation."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "tests/oracles/gen1"))
sys.path.insert(0, str(ROOT / "tests"))
from stochastic_f_common_v2 import (  # noqa: E402
    check_calibration_formal_separation,
)


def planned_purposes(policy, procedure_hash, calibration=False):
    fixtures = policy["calibration"] if calibration else policy["fixtures"]
    result = []
    for fixture in fixtures:
        if not isinstance(fixture, dict):
            raise ValueError("complete frozen calibration tuple required")
        base = (
            fixture["key_base"]
            if "key_base" in fixture
            else {
                "family": "A2",
                "fixture_id": fixture["fixture_id"],
                "frequency_hz": fixture["inputs"]["frequency_hz"],
                "target_id": fixture["target_id"],
                "observable": "acceleration_csd",
                "set": "science",
            }
        )
        if calibration:
            base = {**base, "set": "science"}
        for representation in ("scalar",) if calibration else ("Gfree", "SI", "scalar"):
            scalar = representation == "scalar"
            key = {**base, "representation": "Gfree" if scalar else representation}
            if scalar:
                key.update(
                    target_id="nominal_x",
                    observable="isotropic_depth_factor",
                    set="auxiliary",
                )
            inputs = (
                {
                    "nominal_exact_x": fixture["nominal_exact_x"],
                    "frequency_hz": fixture["inputs"]["frequency_hz"],
                }
                if scalar
                else fixture["inputs"]
            )
            result.append(
                {
                    "purpose": "calibration" if calibration else "formal-science",
                    "set": "calibration" if calibration else key["set"],
                    "key": key,
                    "scientific_input": {
                        "frequency_hz": key["frequency_hz"],
                        "frame": "dimensionless scalar x"
                        if scalar
                        else fixture["frame"],
                        "psd_convention": "dimensionless scalar"
                        if scalar
                        else fixture["psd_convention"],
                        "source_pins": policy["source_pins"],
                        "procedure_hash": procedure_hash,
                        "inputs": inputs,
                    },
                }
            )
    return result


def separation(policy, procedure_hash):
    calibration = planned_purposes(policy, procedure_hash, calibration=True)
    formal = planned_purposes(policy, procedure_hash)
    decision = check_calibration_formal_separation(calibration, formal)
    if not decision["admitted"]:
        raise ValueError("frozen calibration/formal purpose or input overlap")
    return {"calibration": calibration, "formal": formal, "decision": decision}
