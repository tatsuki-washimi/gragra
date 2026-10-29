"""Synthetic connection checks only; no production APIs or formal R artifacts."""

from copy import deepcopy
from decimal import Decimal, localcontext

import mpmath as mp
import pytest

from tests.oracles.gen1 import stochastic_acceptance as kernel
from tests.oracles.gen1.stochastic_formal.common import (
    family_connections_acoustic as plugin,
)
from tests.oracles.gen1.stochastic_formal.common.transport import (
    GateError,
    artifact_pin,
    canonical_bytes,
    key_id,
)


def fixture_context(tmp_path, family="A1", zero=False):
    inputs = {
        "frequency_hz": "2e+0",
        "sound_speed_m_s": "5e+0",
        "pressure_psd_pa2_hz": "3e+0",
        "g_si": "1e-2",
    }
    case = {
        "fixture_id": "synthetic-only",
        "inputs": inputs,
        "pressure_convention": "surface_total" if family == "A1" else "cavern",
    }
    if family == "A3":
        inputs["radius_m"] = "0e+0" if zero else "3e-1"
    with mp.workdps(70):
        base = mp.mpf("0.03")
        if family == "A3":
            x = 2 * mp.pi * 2 * mp.mpf(inputs["radius_m"]) / 5
            factor = (1 - mp.sin(x) / x) if x else mp.mpf(0)
            values = [4 * base * factor**2 / 3] * 3
        else:
            values = list(map(mp.mpf, ["0.01", "0.01", "0.02"]))
        quantities = {}
        references = {}
        keys = {}
        for representation in ("Gfree", "SI"):
            key = {
                "family": family,
                "fixture_id": case["fixture_id"],
                "frequency_hz": inputs["frequency_hz"],
                "observable": "acceleration_csd",
                "representation": representation,
                "target_id": "origin",
                "set": "science",
            }
            data = [{"real": "0e+0", "imag": "0e+0"} for _ in range(9)]
            for index, value in zip((0, 4, 8), values, strict=True):
                scaled = value * (mp.mpf("1e-4") if representation == "SI" else 1)
                with localcontext() as dc:
                    dc.prec = 70
                    data[index]["real"] = plugin._encode(Decimal(mp.nstr(scaled, 60)))
            q = {"shape": [3, 3], "data": data}
            u = {
                "kind": "exact_zero" if zero else "estimate",
                "certificate": "synthetic empty-radius whole observable"
                if zero
                else None,
                "components": {
                    n: {
                        "value": "0e+0" if zero else "1e-40",
                        "method": "synthetic-only allowance",
                    }
                    for n in kernel.COMPONENTS
                },
            }
            quantities[key_id(key)] = q
            references[key_id(key)] = {
                "key": key,
                "inputs": deepcopy(inputs),
                "quantity": q,
                "uncertainty": u,
            }
            keys[representation] = key
    diagnostics = [
        {
            "fixture_id": case["fixture_id"],
            "representation": r,
            "precision_dps": [60, 80],
            "resolutions": [1, 2, 4],
            "alternate_gate": True,
            "alternate_disagreement": "0e+0",
        }
        for r in keys
    ]
    path = tmp_path / "synthetic-diagnostics.json"
    path.write_bytes(canonical_bytes(diagnostics))
    calls = []

    def actual(key):
        identity = key_id(key)
        assert identity in quantities
        calls.append(identity)
        return deepcopy(quantities[identity])

    key = keys["Gfree"]
    return {
        "key": key,
        "reference": references[key_id(key)],
        "production": deepcopy(quantities[key_id(key)]),
        "actual": actual,
        "all_references": references,
        "frozen_fixture": {"science": [case]},
        "frozen_procedure": {
            "connection_checks": list(plugin.CHECKS[family]),
            "claim": (
                "published centered half-spherical cavern; no boundary-acoustics claim"
            ),
        },
        "frozen_budget": {"absolute_rate": "1e-13", "relative_rate": "1e-12"},
        "source_scales": {
            key_id(k): "0e+0" if zero else ("3e-2" if r == "Gfree" else "3e-6")
            for r, k in keys.items()
        },
        "kernel": kernel,
        "reference_artifacts": {
            "method-diagnostics.json": artifact_pin(path, "sealed_reference")
        },
        "calls": calls,
        "quantities": quantities,
    }


@pytest.mark.parametrize("family", ["A1", "A3"])
@pytest.mark.parametrize("representation", ["Gfree", "SI"])
def test_synthetic_frozen_key_connections_pass(tmp_path, family, representation):
    context = fixture_context(tmp_path, family)
    if representation == "SI":
        key = dict(context["key"], representation="SI")
        context.update(
            key=key,
            reference=context["all_references"][key_id(key)],
            production=deepcopy(context["quantities"][key_id(key)]),
        )
    result = plugin.assess(context)
    assert set(result) == {"gates", "evidence"}
    assert set(result["gates"]) == set(plugin.CHECKS[family])
    assert all(value is True for value in result["gates"].values())
    assert len(context["calls"]) == 1


def test_pressure_incident_psd_misnormalization_fails(tmp_path):
    context = fixture_context(tmp_path)
    for entry in context["production"]["data"]:
        entry["real"] = plugin._encode(Decimal(entry["real"]) / 4)
    result = plugin.assess(context)
    assert result["gates"]["pressure normalization amplitude2/PSD4"] is False
    assert result["gates"]["positive-side Green limit"] is False


def test_vertical_component_relation_is_not_default_pass(tmp_path):
    context = fixture_context(tmp_path)
    context["production"]["data"][8]["real"] = "1e-2"
    assert plugin.assess(context)["gates"]["component/total relation"] is False


def test_wrong_g_squared_counterpart_fails(tmp_path):
    context = fixture_context(tmp_path)
    si_key = key_id(dict(context["key"], representation="SI"))
    context["quantities"][si_key]["data"][0]["real"] = "1e-3"
    assert plugin.assess(context)["gates"]["G2 transport"] is False


def test_empty_radius_whole_zero_and_nonzero_counterfeit(tmp_path):
    context = fixture_context(tmp_path, "A3", zero=True)
    assert all(plugin.assess(context)["gates"].values())
    context["production"]["data"][0]["real"] = "1e-30"
    assert (
        plugin.assess(context)["gates"]["independent spherical Newton volume prefactor"]
        is False
    )


def test_direction_average_detects_anisotropy(tmp_path):
    context = fixture_context(tmp_path, "A3")
    context["production"]["data"][4]["real"] = "1e-3"
    assert plugin.assess(context)["gates"]["direction identity average"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "tampered",
        "alternate_false",
        "disagreement",
        "unknown_fixture",
        "unknown_contract",
        "wrong_tuple",
    ],
)
def test_missing_or_counterfeit_evidence_fails_closed(tmp_path, mutation):
    context = fixture_context(tmp_path, "A3")
    pin = context["reference_artifacts"]["method-diagnostics.json"]
    if mutation == "missing":
        context["reference_artifacts"] = {}
    elif mutation == "tampered":
        from pathlib import Path

        Path(pin["path"]).write_text("[]")
    elif mutation in {"alternate_false", "disagreement", "unknown_fixture"}:
        import json
        from pathlib import Path

        path = Path(pin["path"])
        rows = json.loads(path.read_text())
        rows[0][
            {
                "alternate_false": "alternate_gate",
                "disagreement": "alternate_disagreement",
                "unknown_fixture": "fixture_id",
            }[mutation]
        ] = {
            "alternate_false": False,
            "disagreement": "1e-20",
            "unknown_fixture": "other",
        }[mutation]
        path.write_bytes(canonical_bytes(rows))
        context["reference_artifacts"]["method-diagnostics.json"] = artifact_pin(
            path, "sealed_reference"
        )
        if mutation != "unknown_fixture":
            assert (
                plugin.assess(context)["gates"]["radial/angular kernel agreement"]
                is False
            )
            return
    elif mutation == "unknown_contract":
        context["frozen_procedure"]["connection_checks"].append("unreviewed")
    else:
        context["reference"]["inputs"]["radius_m"] = "1e+0"
    with pytest.raises((GateError, ValueError)):
        plugin.assess(context)


@pytest.mark.parametrize("x", ["0.0041", "0.63", "3.7", "45.9"])
def test_decimal_volume_identity_against_independent_angular_integral(x):
    with localcontext() as dc, mp.workdps(90):
        dc.prec = 90
        actual = plugin._volume_kernel(Decimal(x))
        expected = mp.quad(lambda u: 2 * mp.sin(mp.mpf(x) * u / 2) ** 2, [0, 1])
        assert abs(mp.mpf(str(actual)) - expected) < mp.mpf("1e-75")
