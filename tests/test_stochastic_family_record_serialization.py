"""Full synthetic evaluator records must survive the admitted strict transport."""

import importlib
import types
from fractions import Fraction

import numpy as np
import pytest

from tests.oracles.gen1.stochastic_formal.common import dependencies
from tests.oracles.gen1.stochastic_formal.common import family_adapter as adapter
from tests.oracles.gen1.stochastic_formal.common.transport import (
    artifact_pin,
    canonical_bytes,
    key_id,
    strict_json,
)


@pytest.mark.parametrize(
    "family,scalar_auxiliary",
    [
        ("R1", False),
        ("B2", False),
        ("A1", False),
        ("A3", False),
        ("T1", False),
        ("A2", False),
        ("A2", True),
    ],
)
def test_full_evaluator_record_preserves_complete_call_keys(
    tmp_path, monkeypatch, family, scalar_auxiliary
):
    scalar = family == "T1" or scalar_auxiliary
    key = {
        "family": family,
        "fixture_id": "synthetic-serialization-only",
        "frequency_hz": "2e+0",
        "observable": "isotropic_depth_factor"
        if scalar_auxiliary
        else ("projected_acceleration_psd" if family == "T1" else "acceleration_csd"),
        "representation": "Gfree",
        "target_id": "nominal_x" if scalar_auxiliary else "origin",
        "set": "auxiliary" if scalar_auxiliary else "science",
    }
    keys = [key] if scalar_auxiliary else [key, {**key, "representation": "SI"}]
    shape = [] if scalar else [3, 3]
    connector = types.ModuleType("synthetic_serialization_connector")
    connector.__file__ = str(tmp_path / "connector.py")
    (tmp_path / "connector.py").write_text("# synthetic only\n")
    connector.assess = lambda _context: {
        "gates": {"synthetic_obligation": True},
        "evidence": {"synthetic": True},
    }
    plans = {
        key_id(k): {
            "module": "synthetic_api",
            "symbol": k["representation"],
            "args": [],
            "kwargs": {"frequency_hz": "2e+0"},
            "shape": shape,
            "output_index": [],
        }
        for k in keys
    }

    def value(frequency_hz, *, factor=1.0):
        assert frequency_hz == 2.0
        return np.float64(10 * factor) if scalar else np.eye(3) * (10 * factor)

    api = types.SimpleNamespace(
        Gfree=value, SI=lambda frequency_hz: value(frequency_hz, factor=0.1**2)
    )
    real_import = importlib.import_module

    def synthetic_import(name, *args, **kwargs):
        if name == connector.__name__:
            return connector
        if name == "synthetic_api":
            return api
        if name == "gragra.constants":
            return types.SimpleNamespace(G_SI=0.1)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(adapter.importlib, "import_module", synthetic_import)
    monkeypatch.setattr(dependencies, "verify_runtime_closure", lambda *_args: None)
    kernel = types.SimpleNamespace(
        promote_binary64=repr,
        decimal_fraction=Fraction,
        fraction_decimal=lambda value, *_args, **_kwargs: str(float(value)),
        structure_gates=lambda *_args: {"synthetic_structure": True},
        assess=lambda *_args: {"accepted": True},
    )
    records = {
        key_id(k): {
            "key": k,
            "inputs": {"g_si": "1e-1"},
            "quantity": {"shape": shape},
            "uncertainty": {},
            "units": "1" if scalar_auxiliary else "synthetic-unit",
            "status": "resolved",
        }
        for k in keys
    }
    prepared = {
        "kernel": kernel,
        "plans": plans,
        "records": records,
        "policy": {
            "connection_assessor": {
                "module": connector.__name__,
                "symbol": "assess",
                "source": artifact_pin(connector.__file__, "source"),
            },
            "connection_requirements": ["synthetic_obligation"],
        },
        "pins": {"production_commit": "a" * 40, "production_tree": "b" * 40},
        "closure": {"modules": []},
        "budget": {"absolute_rate": "0", "relative_rate": "1"},
        "scales": {key_id(k): "1" for k in keys},
        "source_review": {"packet": {}, "pins": {}},
        "reference_artifacts": {},
        "fixtures": {
            "science": [{"fixture_id": key["fixture_id"]}],
            "units": {"Gfree": "synthetic-unit", "SI": "synthetic-unit"},
        },
        "procedure": {},
        "record_hashes": {key_id(k): "c" * 64 for k in keys},
    }
    evaluator = adapter.production_evaluator(prepared)
    record = strict_json(canonical_bytes(evaluator(key)))
    assert record["passed"] is True
    assert isinstance(record["actual_calls"], list)
    assert {key_id(row["key"]) for row in record["actual_calls"]} == {
        key_id(k) for k in keys
    }
    assert [key_id(row["key"]) for row in record["actual_calls"]] == sorted(
        key_id(k) for k in keys
    )
    assert all(set(row) == {"key", "call"} for row in record["actual_calls"])
    assert all(
        row["call"]["module"] == "synthetic_api" for row in record["actual_calls"]
    )


@pytest.mark.parametrize("family", ["R1", "B2", "A1", "A3", "T1"])
def test_real_family_plugin_synthetic_evidence_is_canonical(tmp_path, family):
    if family in {"R1", "B2"}:
        source = importlib.import_module(
            "tests.test_stochastic_family_connections_seismic"
        )
        context = (source.numeric_context if family == "R1" else source.b2_context)(
            tmp_path
        )
        result = source.plugin().assess(context)
    elif family in {"A1", "A3"}:
        source = importlib.import_module(
            "tests.test_stochastic_family_connections_acoustic"
        )
        result = source.plugin.assess(source.fixture_context(tmp_path, family))
    else:
        source = importlib.import_module(
            "tests.test_stochastic_family_connections_temperature"
        )
        result = source.module().assess(source.context(tmp_path))
    assert (
        strict_json(canonical_bytes({"connection_assessment": result}))[
            "connection_assessment"
        ]
        == result
    )
