"""Synthetic source-packet checks; no formal reference or production calls."""

import importlib
from copy import deepcopy

import pytest

from tests.oracles.gen1.stochastic_formal.common.transport import (
    GateError,
    artifact_pin,
    canonical_bytes,
)

MODULE = "tests.oracles.gen1.stochastic_formal.common.family_connections_seismic"


def plugin():
    return importlib.import_module(MODULE)


def packet(tmp_path, family="R1"):
    p = plugin()
    pins = {
        "family": family,
        "production_commit": "a" * 40,
        "production_tree": "b" * 40,
        "frozen_artifacts_sha256": {"F": "c" * 64},
        "reference_seal_sha256": "d" * 64,
        "reference_records_sha256": "e" * 64,
        "expected_keys_sha256": "f" * 64,
        "source_files": {"src/private.py": "0" * 64},
    }
    proof = tmp_path / "proof.json"
    proof.write_bytes(canonical_bytes({"source_only": True}))
    reviews = {}
    for role in ("science", "numerical", "provenance"):
        report = tmp_path / (role + "-original.json")
        report.write_bytes(canonical_bytes({"review": role}))
        claims = (
            {
                claim: {
                    "decision": "PASS",
                    "scope": "SOURCE_MODEL_BINDING_ONLY",
                    "proof_report": artifact_pin(proof, "source_proof"),
                }
                for claim in p.CHECKS[family]
            }
            if role == "science"
            else {}
        )
        value = {
            "schema": "gen1-stochastic-production-source-review-v1",
            "family": family,
            "role": role,
            "reviewer_id": role + "-independent",
            "decision": "PASS",
            "pins": pins,
            "claim_attestations": claims,
            "original_review": artifact_pin(report, "source_review"),
        }
        path = tmp_path / (role + ".json")
        path.write_bytes(canonical_bytes(value))
        reviews[role] = artifact_pin(path, "production_source_review")
    admission = tmp_path / "admission.json"
    admission.write_bytes(
        canonical_bytes(
            {
                "schema": "gen1-stochastic-production-source-admission-v1",
                "family": family,
                "decision": "PASS",
                "pins": pins,
                "reviews": reviews,
            }
        )
    )
    return {
        "production_source_pins": pins,
        "production_source_reviews": {
            "admission": artifact_pin(admission, "production_source_admission"),
            "reviews": reviews,
        },
    }


def test_source_packet_requires_actual_three_role_proofs(tmp_path):
    context = packet(tmp_path)
    result = plugin().verify_source_claims(context, "R1")
    assert set(result) == set(plugin().CHECKS["R1"])
    assert all(result.values())


@pytest.mark.parametrize(
    "mutation", ["missing", "stale", "counterfeit", "scope", "duplicate"]
)
def test_source_packet_defects_fail_closed(tmp_path, mutation):
    import json
    from pathlib import Path

    context = packet(tmp_path)
    if mutation == "missing":
        del context["production_source_reviews"]["reviews"]["science"]
    elif mutation == "stale":
        context["production_source_pins"] = deepcopy(context["production_source_pins"])
        context["production_source_pins"]["production_commit"] = "1" * 40
    elif mutation == "counterfeit":
        Path(
            context["production_source_reviews"]["reviews"]["science"]["path"]
        ).write_text("{}")
    else:
        pin = context["production_source_reviews"]["reviews"]["science"]
        path = Path(pin["path"])
        value = json.loads(path.read_bytes())
        if mutation == "scope":
            next(iter(value["claim_attestations"].values()))["scope"] = "NUMERIC_PASS"
        else:
            value["reviewer_id"] = "numerical-independent"
        path.write_bytes(canonical_bytes(value))
        context["production_source_reviews"]["reviews"]["science"] = artifact_pin(
            path, pin["kind"]
        )
        admission_pin = context["production_source_reviews"]["admission"]
        admission_path = Path(admission_pin["path"])
        admission = json.loads(admission_path.read_bytes())
        admission["reviews"] = context["production_source_reviews"]["reviews"]
        admission_path.write_bytes(canonical_bytes(admission))
        context["production_source_reviews"]["admission"] = artifact_pin(
            admission_path, admission_pin["kind"]
        )
    with pytest.raises((GateError, KeyError)):
        plugin().verify_source_claims(context, "R1")


def numeric_context(tmp_path):
    from tests.oracles.gen1 import stochastic_acceptance as kernel
    from tests.oracles.gen1.stochastic_formal.common.transport import key_id

    p = plugin()
    context = packet(tmp_path)
    case = {
        "fixture_id": "synthetic-only",
        "frequency_hz": ["2e+0"],
        "height_m": "1e+0",
        "density_kg_m3": "2e+0",
        "v_p_m_s": "6e+0",
        "v_s_m_s": "3e+0",
        "g_si": "1e-2",
        "surface_vertical_displacement_psd_m2_hz": "1e-3",
        "frame": "Cartesian surface frame z up",
        "psd_convention": "one-sided per-Hz",
        "input_assumptions": "exact_input",
        "target_id": "height",
        "set": "science",
    }
    quantities = {}
    refs = {}
    keys = []
    for representation in ("Gfree", "SI"):
        key = {
            "family": "R1",
            "fixture_id": case["fixture_id"],
            "frequency_hz": "2e+0",
            "target_id": "height",
            "observable": "acceleration_csd",
            "representation": representation,
            "set": "science",
        }
        factor = "1e-2" if representation == "Gfree" else "1e-6"
        data = [{"real": "0e+0", "imag": "0e+0"} for _ in range(9)]
        data[0]["real"] = data[4]["real"] = factor
        data[8]["real"] = "2e-2" if representation == "Gfree" else "2e-6"
        q = {"shape": [3, 3], "data": data}
        quantities[key_id(key)] = q
        keys.append(key)
        inputs = dict(case)
        inputs["frequency_hz"] = key["frequency_hz"]
        refs[key_id(key)] = {
            "key": key,
            "inputs": inputs,
            "quantity": q,
            "uncertainty": {
                "kind": "estimate",
                "certificate": None,
                "components": {
                    n: {"value": "1e-40", "method": "synthetic"}
                    for n in kernel.COMPONENTS
                },
            },
        }
    diagnostics = [
        {
            "key": k,
            "status": "resolved",
            "method": {
                "rayleigh_speed_m_s": "2.8e+0",
                "horizontal_literal": "1e-2",
                "literal_difference_norm": "0e+0",
                "precision_digits": [60, 90],
                "orders": [24, 48, 96],
            },
        }
        for k in keys
    ]
    path = tmp_path / "diagnostics.json"
    path.write_bytes(canonical_bytes(diagnostics))
    context.update(
        key=keys[0],
        reference=refs[key_id(keys[0])],
        production=deepcopy(quantities[key_id(keys[0])]),
        actual=lambda k: deepcopy(quantities[key_id(k)]),
        all_references=refs,
        frozen_fixture={"science": [case], "expected_keys": keys},
        frozen_procedure={"required_connections": list(p.CHECKS["R1"])},
        frozen_budget={"absolute_rate": "1e-13", "relative_rate": "1e-12"},
        kernel=kernel,
        source_scales={key_id(k): "1e+0" for k in keys},
        reference_artifacts={
            "reference-diagnostics.json": artifact_pin(path, "sealed_reference")
        },
    )
    return context


def test_finite_r1_full_claims_require_attenuation_peer(tmp_path):
    context = numeric_context(tmp_path)
    result = plugin().assess(context)
    assert result["gates"][plugin().CHECKS["R1"][1]] is True
    assert result["gates"][plugin().CHECKS["R1"][2]] is True
    assert result["gates"][plugin().CHECKS["R1"][3]] is False
    assert set(result) == {"gates", "evidence"}


def test_anisotropy_or_offdiagonal_cannot_pass(tmp_path):
    context = numeric_context(tmp_path)
    context["production"]["data"][1]["real"] = "1e-3"
    result = plugin().assess(context)
    assert result["gates"][plugin().CHECKS["R1"][1]] is False
    assert result["gates"][plugin().CHECKS["R1"][4]] is False


def test_r1_actual_finite_height_pair_and_wrong_attenuation(tmp_path):
    from decimal import Decimal, localcontext

    context = numeric_context(tmp_path)
    with localcontext() as dc:
        dc.prec = 100
        from tests.oracles.gen1.stochastic_formal.common.transport import key_id

        speed = Decimal("2.8")
        q = (1 - (speed / 6) ** 2).sqrt()
        zeta = (q / (1 - (speed / 3) ** 2).sqrt()).sqrt()
        h = (
            2
            * (plugin().PI * 2 * (1 - zeta) / (q - zeta)) ** 2
            * Decimal("1e-3")
            * (-4 * plugin().PI * 2 / speed).exp()
        )
        base = plugin()._quantity(
            [
                (h * (2 if i == 8 else 1) if i in (0, 4, 8) else Decimal(0), Decimal(0))
                for i in range(9)
            ]
        )
        context["production"] = base
        context["reference"]["quantity"] = base
        si = dict(context["key"], representation="SI")
        context["all_references"][key_id(si)]["quantity"] = plugin()._quantity(
            [(a * Decimal("1e-4"), b) for a, b in plugin()._values(base)]
        )
        old = context["actual"]
        context["actual"] = lambda k: (
            context["all_references"][key_id(si)]["quantity"] if k == si else old(k)
        )
        case = deepcopy(context["frozen_fixture"]["science"][0])
        case.update(fixture_id="synthetic-height-peer", height_m="2e+0")
        context["frozen_fixture"]["science"].append(case)
        peer = dict(context["key"], fixture_id=case["fixture_id"])
        context["frozen_fixture"]["expected_keys"].append(peer)
        exponent = -4 * plugin().PI * 2 / Decimal("2.8")
        values = [
            (a * exponent.exp(), b) for a, b in plugin()._values(context["production"])
        ]
        q = plugin()._quantity(values)
        original = context["actual"]
        context["actual"] = lambda k: q if k == peer else original(k)
        assert plugin().assess(context)["gates"][plugin().CHECKS["R1"][3]] is True
        context["actual"] = lambda k: (
            context["production"] if k == peer else original(k)
        )
        assert plugin().assess(context)["gates"][plugin().CHECKS["R1"][3]] is False


def test_series_small_limit_and_intermediate_identity():
    from decimal import Decimal, localcontext

    import mpmath as mp

    with localcontext() as dc, mp.workdps(100):
        dc.prec = 100
        for x in ("0", "1e-6", "0.63", "9.3", "19.9"):
            result = plugin()._factor(Decimal(x))
            z = mp.mpf(x)
            expected = (mp.sin(z) - z * mp.cos(z)) / z**3 if z else mp.mpf(1) / 3
            assert abs(mp.mpf(str(result)) - expected) < mp.mpf("1e-80")
    with pytest.raises(GateError):
        plugin()._factor(Decimal("21"))


def b2_context(tmp_path):
    from decimal import Decimal, localcontext

    from tests.oracles.gen1 import stochastic_acceptance as kernel
    from tests.oracles.gen1.stochastic_formal.common.transport import key_id

    context = packet(tmp_path, "B2")
    rows = []
    keys = []
    refs = {}
    quantities = {}
    diagnostics = []
    with localcontext() as dc:
        dc.prec = 100
        for eta in ("0e+0", "2e-1", "5e-1", "8e-1", "1e+0"):
            e = Decimal(eta)
            case = {
                "fixture_id": "synthetic-eta-" + eta,
                "frequency_hz": ["2e+0"],
                "density_kg_m3": "2e+0",
                "radius_m": "3e-1",
                "v_p_m_s": "6e+0",
                "v_s_m_s": "3e+0",
                "eta_P": eta,
                "p_displacement_psd_m2_hz": plugin()._encode(e * Decimal("1e-3")),
                "s_displacement_psd_m2_hz": plugin()._encode((1 - e) * Decimal("1e-3")),
                "p_s_cross_csd": "0e+0",
                "elastic_scattering": "excluded",
                "g_si": "1e-2",
                "frame": "Cartesian cavity center",
                "psd_convention": "one-sided per-Hz",
                "input_assumptions": "exact_input",
                "target_id": "center",
                "set": "science",
            }
            rows.append(case)
            for rep in ("Gfree", "SI"):
                k = {
                    "family": "B2",
                    "fixture_id": case["fixture_id"],
                    "frequency_hz": "2e+0",
                    "representation": rep,
                    "target_id": "center",
                    "observable": "acceleration_csd",
                    "set": "science",
                }
                keys.append(k)
                inputs = dict(case)
                inputs["frequency_hz"] = "2e+0"
                d = (4 * plugin().PI * 2) ** 2 * (
                    4
                    * e
                    * Decimal("1e-3")
                    * plugin()._factor(2 * plugin().PI * 2 * Decimal(".3") / 6) ** 2
                    + (1 - e)
                    * Decimal("1e-3")
                    * plugin()._factor(2 * plugin().PI * 2 * Decimal(".3") / 3) ** 2
                )
                if rep == "SI":
                    d *= Decimal("1e-4")
                q = plugin()._quantity(
                    [
                        (d if i in (0, 4, 8) else Decimal(0), Decimal(0))
                        for i in range(9)
                    ]
                )
                quantities[key_id(k)] = q
                refs[key_id(k)] = {
                    "key": k,
                    "inputs": inputs,
                    "quantity": q,
                    "uncertainty": {
                        "kind": "estimate",
                        "certificate": None,
                        "components": {
                            n: {"value": "1e-40", "method": "synthetic"}
                            for n in kernel.COMPONENTS
                        },
                    },
                }
                diagnostics.append(
                    {
                        "key": k,
                        "status": "resolved",
                        "method": {
                            "literal_difference_norm": "0e+0",
                            "precision_digits": [60, 90],
                            "orders": [24, 48, 96],
                        },
                    }
                )
    path = tmp_path / "b2-diagnostics.json"
    path.write_bytes(canonical_bytes(diagnostics))
    key = keys[4]
    context.update(
        key=key,
        reference=refs[key_id(key)],
        production=deepcopy(quantities[key_id(key)]),
        actual=lambda k: deepcopy(quantities[key_id(k)]),
        all_references=refs,
        frozen_fixture={"science": rows, "expected_keys": keys},
        frozen_procedure={"required_connections": list(plugin().CHECKS["B2"])},
        frozen_budget={"absolute_rate": "1e-13", "relative_rate": "1e-12"},
        kernel=kernel,
        source_scales={key_id(k): "1e+0" for k in keys},
        reference_artifacts={
            "reference-diagnostics.json": artifact_pin(path, "sealed_reference")
        },
    )
    return context


def test_b2_full_incoherent_mixture_and_spherical_claims(tmp_path):
    context = b2_context(tmp_path)
    assert all(plugin().assess(context)["gates"].values())
    context["production"]["data"][0]["real"] = "1e+0"
    result = plugin().assess(context)
    assert result["gates"][plugin().CHECKS["B2"][0]] is False
    assert result["gates"][plugin().CHECKS["B2"][1]] is False


def test_b2_missing_pure_endpoint_cannot_pass(tmp_path):
    context = b2_context(tmp_path)
    context["frozen_fixture"]["science"] = context["frozen_fixture"]["science"][:-1]
    assert plugin().assess(context)["gates"][plugin().CHECKS["B2"][0]] is False


def test_b2_missing_interior_eta_cannot_claim_complete_coverage(tmp_path):
    context = b2_context(tmp_path)
    context["frozen_fixture"]["science"] = [
        r for r in context["frozen_fixture"]["science"] if r["eta_P"] != "8e-1"
    ]
    assert plugin().assess(context)["gates"][plugin().CHECKS["B2"][0]] is False


@pytest.mark.parametrize(
    "mutation",
    ["diagnostic_missing", "unknown_contract", "cross_nonzero", "scattering"],
)
def test_frozen_seismic_contract_failures(tmp_path, mutation):
    context = b2_context(tmp_path)
    if mutation == "diagnostic_missing":
        context["reference_artifacts"] = {}
    elif mutation == "unknown_contract":
        context["frozen_procedure"]["required_connections"].append("new scope")
    else:
        name = "p_s_cross_csd" if mutation == "cross_nonzero" else "elastic_scattering"
        value = "1e-3" if mutation == "cross_nonzero" else "included"
        case = context["frozen_fixture"]["science"][2]
        case[name] = value
        context["reference"]["inputs"][name] = value
    with pytest.raises((GateError, KeyError)):
        plugin().assess(context)


def test_cartesian_source_psds_cancel_angular_thirds():
    """Independent quadrature of 3SP longitudinal and 1.5SS/mode transverse."""
    import mpmath as mp

    with mp.workdps(80):
        # x-direction squared P projection averages1/3 on the whole sphere.
        p_mean = (
            mp.quad(lambda t: mp.sin(t) ** 3, [0, mp.pi])
            * mp.quad(lambda a: mp.cos(a) ** 2, [0, 2 * mp.pi])
            / (4 * mp.pi)
        )
        # Two orthonormal S polarizations sum I-nn^T, not another P projector.
        s_mean = 1 - p_mean
        assert abs(3 * p_mean - 1) < mp.mpf("1e-75")
        assert abs(mp.mpf("1.5") * s_mean - 1) < mp.mpf("1e-75")
        # With longitudinal amplitude2J and transverse amplitudeJ, the full
        # per-Cartesian source result is4SP*Jp²+SS*Js², without another third.
        sp, ss, jp, js = map(mp.mpf, ("0.00023", "0.00077", "0.31", "0.29"))
        directional = (
            3 * sp * (2 * jp) ** 2 * p_mean + mp.mpf("1.5") * ss * js**2 * s_mean
        )
        assert abs(directional - (4 * sp * jp**2 + ss * js**2)) < mp.mpf("1e-75")


def test_full_sealed_fixture_inputs_cannot_be_replaced_by_physical_subset(tmp_path):
    context = numeric_context(tmp_path)
    context["reference"]["inputs"].pop("frame")
    with pytest.raises(GateError):
        plugin().assess(context)
