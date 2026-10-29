"""Reviewed family wrapper over immutable common supervisor/sealer.

--worker emits production-blind candidates. Formal supervisor requires exact
review receipt and durable E manifest; candidate computation is not admission.
"""

import argparse
import hashlib
import importlib.abc
import json
import os
import platform
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[3]))


class DenyProduction(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):  # noqa: ARG002
        if fullname == "gragra" or fullname.startswith("gragra."):
            raise ImportError("formal reference denies production imports")
        return


if any(n == "gragra" or n.startswith("gragra.") for n in sys.modules):
    raise RuntimeError("production module already loaded before reference import")
sys.meta_path.insert(0, DenyProduction())
import generator  # noqa: E402
from oracles.gen1 import stochastic_acceptance as acceptance  # noqa: E402
from oracles.gen1 import stochastic_f_common_v2 as common_v2  # noqa: E402
from run_stochastic_local import (  # noqa: E402
    run_frozen_attempt,
    verify_environment_inventory,
    verify_source_pin,
)
from stochastic_attempt import canonical_json_bytes  # noqa: E402

FAMILY = HERE.name.upper()
KEY_FIELDS = [
    "family",
    "fixture_id",
    "frequency_hz",
    "target_id",
    "observable",
    "representation",
    "set",
]


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON property")
        result[key] = value
    return result


def reject_number(value):
    raise ValueError("nonfinite or floating scientific JSON number: " + value)


def parse_json(raw):
    return json.loads(
        raw,
        object_pairs_hook=unique_pairs,
        parse_float=reject_number,
        parse_constant=reject_number,
    )


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory():
    platform.python_version()  # Load version helper before the stable snapshot.
    return {
        "python_executable": sys.executable,
        "python_binary_sha256": digest(sys.executable),
        "python_version": sys.version,
        "mpmath_version": generator.mp.__version__,
        "generator_sha256": digest(HERE / "generator.py"),
        "runner_sha256": digest(__file__),
        "production_modules": sorted(
            n for n in sys.modules if n == "gragra" or n.startswith("gragra.")
        ),
        "loaded_modules": {
            n: {
                "path": getattr(m, "__file__", None),
                "sha256": digest(m.__file__)
                if getattr(m, "__file__", None) and Path(m.__file__).is_file()
                else None,
            }
            for n, m in sys.modules.copy().items()
        },
    }


def check_runtime(expected_path):
    observed = inventory()
    if observed["production_modules"]:
        raise RuntimeError("production module already loaded")
    if observed != parse_json(Path(expected_path).read_bytes()):
        raise RuntimeError("runtime environment mismatch before computation")
    return observed


def physical_inputs(inputs):
    fields = (
        (
            "density_kg_m3",
            "v_p_m_s",
            "v_s_m_s",
            "frequency_hz",
            "g_si",
            "height_m",
            "surface_vertical_displacement_psd_m2_hz",
        )
        if FAMILY == "R1"
        else (
            "density_kg_m3",
            "v_p_m_s",
            "v_s_m_s",
            "frequency_hz",
            "g_si",
            "radius_m",
            "p_displacement_psd_m2_hz",
            "s_displacement_psd_m2_hz",
            "p_s_cross_csd",
        )
    )
    result = {k: acceptance.canonical_decimal(Decimal(inputs[k])) for k in fields}
    if FAMILY == "B2":
        result["elastic_scattering"] = inputs.get("elastic_scattering", "excluded")
    return result


def purpose_metadata(key, inputs, pins, procedure_hash, purpose="formal-science"):
    metadata = {
        "purpose": purpose,
        "set": "calibration" if purpose == "calibration" else key["set"],
        "key": key,
        "scientific_input": {
            "frequency_hz": key["frequency_hz"],
            "frame": inputs["frame"],
            "psd_convention": inputs["psd_convention"],
            "source_pins": pins,
            "procedure_hash": procedure_hash,
            "inputs": physical_inputs(inputs),
        },
    }
    common_v2.validate_purpose_record(metadata)
    common_v2.canonical_scientific_input_hash(metadata["scientific_input"])
    return metadata


def calibration_metadata(fixtures, procedure, pins, procedure_hash):
    result = []
    for fixture in fixtures.get("calibration", []):
        for frequency in fixture["frequency_hz"]:
            inputs = {**fixture, "frequency_hz": frequency, "g_si": "6.6743e-11"}
            key = dict(
                zip(
                    KEY_FIELDS,
                    [
                        FAMILY,
                        fixture["fixture_id"],
                        frequency,
                        fixture["target_id"],
                        "acceleration_csd",
                        "Gfree",
                        "science",
                    ],
                    strict=True,
                )
            )
            result.append(
                purpose_metadata(key, inputs, pins, procedure_hash, "calibration")
            )
    # Preserve the disclosed earlier development h10 evaluation in separation.
    history_hash = procedure.get("development_overlap_history_sha256")
    if history_hash:
        artifact = next(
            a
            for a in procedure["immutable_author_proposals"]
            if a["sha256"] == history_hash
        )
        if digest(artifact["immutable_path"]) != history_hash:
            raise ValueError("development history byte mismatch")
        history = parse_json(Path(artifact["immutable_path"]).read_bytes())
        old = history["evaluated_development_tuple"]
        inputs = {
            "density_kg_m3": old["rho"],
            "v_p_m_s": old["vp"],
            "v_s_m_s": old["vs"],
            "frequency_hz": old["f"],
            "height_m": old["h"],
            "surface_vertical_displacement_psd_m2_hz": old["PSD"],
            "g_si": "6.6743e-11",
            "frame": fixtures["science"][0]["frame"],
            "psd_convention": fixtures["science"][0]["psd_convention"],
        }
        inputs["frequency_hz"] = acceptance.canonical_decimal(
            Decimal(inputs["frequency_hz"])
        )
        key = dict(
            zip(
                KEY_FIELDS,
                [
                    FAMILY,
                    "development_h10",
                    inputs["frequency_hz"],
                    "height",
                    "acceleration_csd",
                    "Gfree",
                    "science",
                ],
                strict=True,
            )
        )
        result.append(
            purpose_metadata(key, inputs, pins, procedure_hash, "calibration")
        )
    return result


def validate_external_attempt(attempt):
    candidate = Path(attempt).resolve()
    if candidate.exists():
        raise FileExistsError("fresh attempt required")
    parent = candidate.parent
    while not parent.exists():
        parent = parent.parent
    result = subprocess.run(
        ["git", "-C", str(parent), "rev-parse", "--show-toplevel"], capture_output=True
    )
    if result.returncode == 0:
        raise ValueError("formal attempt must be outside every Git working tree")


def validate_durable_e(manifest):
    root = Path(manifest["integration_root"]).resolve()
    observed = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if observed != manifest["integration_head"]:
        raise ValueError("integration head differs from durableE pin")
    result = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "merge-base",
            "--is-ancestor",
            manifest["E_commit"],
            observed,
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        raise ValueError("E is not an actual integration ancestor")


def validate_freeze(manifest, fixtures_path, procedure_path):
    artifact = manifest["freeze_declaration"]
    if digest(artifact["path"]) != artifact["sha256"]:
        raise ValueError("freeze declaration byte mismatch")
    declaration = parse_json(Path(artifact["path"]).read_bytes())
    if declaration["status"] != "FROZEN" or declaration["family"] != FAMILY:
        raise ValueError("actual frozen declaration required")
    if declaration["freeze_id"] != "gen1-stochastic-F_" + FAMILY + "-v1":
        raise ValueError("freeze identifier/family mismatch")
    declared_paths = {
        declaration[k + "_file"] for k in ("fixture", "budget", "procedure")
    }
    if set(declaration["artifact_hashes"]) != declared_paths:
        raise ValueError("freeze requires exactly the three declared artifact paths")
    for kind in ("fixture", "budget", "procedure"):
        if (
            declaration[kind + "_sha256"]
            != declaration["artifact_hashes"][declaration[kind + "_file"]]
        ):
            raise ValueError("direct/map freeze hash mismatch")
    reviews = declaration["reviews"]
    if set(reviews) != {"science", "numerical", "provenance"}:
        raise ValueError("three independent protocol reviews required")
    reviewer_ids = set()
    fields = {
        "schema_version",
        "role",
        "reviewer_id",
        "decision",
        "scope",
        "family",
        "reviewed_commit",
        "artifact_hashes",
        "original_reviews",
    }
    for role, review in reviews.items():
        if review["decision"] != "PASS" or digest(review["path"]) != review["sha256"]:
            raise ValueError("protocol review artifact byte/verdict mismatch")
        actual = parse_json(Path(review["path"]).read_bytes())
        if (
            set(actual) != fields
            or actual["schema_version"] != 1
            or actual["decision"] != "PASS"
            or actual["role"] != role
            or actual["scope"] != "protocol freeze"
            or actual["family"] != FAMILY
            or actual["reviewed_commit"] != declaration["reviewed_commit"]
            or actual["artifact_hashes"] != declaration["artifact_hashes"]
            or not actual["reviewer_id"]
            or actual["reviewer_id"] in reviewer_ids
        ):
            raise ValueError(
                "protocol review semantic candidate/artifact/role binding mismatch"
            )
        reviewer_ids.add(actual["reviewer_id"])
        if (
            not isinstance(actual["original_reviews"], list)
            or not actual["original_reviews"]
        ):
            raise ValueError("protocol review original reports required")
        for original in actual["original_reviews"]:
            if (
                set(original) != {"path", "sha256"}
                or digest(original["path"]) != original["sha256"]
            ):
                raise ValueError("original protocol report byte mismatch")
    concurrence = declaration["root_concurrence"]
    decision = "PASS_FOR_F_" + FAMILY + "_V1_FREEZE"
    if (
        concurrence["decision"] != decision
        or digest(concurrence["path"]) != concurrence["sha256"]
    ):
        raise ValueError("root concurrence byte/verdict mismatch")
    receipt = parse_json(Path(concurrence["path"]).read_bytes())
    if (
        receipt["decision"] != decision
        or receipt["family"] != FAMILY
        or receipt["artifact_hashes"] != declaration["artifact_hashes"]
        or receipt["review_hashes"]
        != {role: review["sha256"] for role, review in reviews.items()}
    ):
        raise ValueError("root concurrence family/actual F pin mismatch")
    procedure = parse_json(Path(procedure_path).read_bytes())
    repo = Path(manifest["repository_root"]).resolve()
    if declaration["baseline_pins"] != procedure["baseline_pins"]:
        raise ValueError("freeze baseline contract mismatch")
    for relative, sha in declaration["baseline_pins"].items():
        if digest(repo / relative) != sha:
            raise ValueError("baseline common bytes changed")
    for path in (fixtures_path, procedure_path, str(repo / procedure["budget_file"])):
        relative = str(Path(path).resolve().relative_to(repo))
        if declaration["artifact_hashes"].get(relative) != digest(path):
            raise ValueError("freeze artifact byte mismatch")
    if (
        digest(fixtures_path) != procedure["fixture_sha256"]
        or digest(repo / procedure["budget_file"]) != procedure["budget_sha256"]
    ):
        raise ValueError("procedure fixture/budget byte binding mismatch")


def selected_cases(fixtures, purpose):
    formal_cases = fixtures["science"]
    formal_keys = fixtures["expected_keys"]
    if purpose == "formal-science":
        return formal_cases, formal_keys
    cases = fixtures["calibration"]
    if not cases:
        raise ValueError("declared calibration fixtures required")
    keys = [
        dict(
            zip(
                KEY_FIELDS,
                [
                    FAMILY,
                    c["fixture_id"],
                    frequency,
                    c["target_id"],
                    "acceleration_csd",
                    representation,
                    "science",
                ],
                strict=True,
            )
        )
        for c in cases
        for frequency in c["frequency_hz"]
        for representation in ("Gfree", "SI")
    ]
    return cases, keys


def emit(
    fixtures_path,
    procedure_path,
    metadata_path=None,
    environment_path=None,
    expected_environment=None,
    purpose="formal-science",
    records_path=None,
):
    observed = (
        check_runtime(expected_environment) if expected_environment else inventory()
    )
    fixtures = parse_json(Path(fixtures_path).read_bytes())
    procedure = parse_json(Path(procedure_path).read_bytes())
    cases, expected = selected_cases(fixtures, purpose)
    if len({canonical_json_bytes(k) for k in expected}) != len(expected):
        raise ValueError("duplicate frozen key")
    lookup = {c["fixture_id"]: c for c in cases}
    if len(lookup) != len(cases):
        raise ValueError("duplicate frozen fixture")
    for key in expected:
        if (
            set(key) != set(KEY_FIELDS)
            or key["set"] != "science"
            or key["representation"] not in ("Gfree", "SI")
        ):
            raise ValueError("unsupported frozen key schema/domain")
        if key["family"] != FAMILY or key["observable"] != "acceleration_csd":
            raise ValueError("wrong frozen family/observable")
        fixture = lookup[key["fixture_id"]]
        if (
            key["frequency_hz"] not in fixture["frequency_hz"]
            or key["target_id"] != fixture["target_id"]
        ):
            raise ValueError("fixture/key mismatch")
    source = procedure.get("source", procedure.get("source_pins"))
    if isinstance(source, dict):
        source = [source]
    pins = [{k: s[k] for k in ("sha256", "equations", "pages")} for s in source]
    metadata = []
    prepared = []
    procedure_hash = digest(procedure_path)
    for key in expected:
        inputs = {
            **lookup[key["fixture_id"]],
            "frequency_hz": key["frequency_hz"],
            "g_si": "6.6743e-11",
        }
        prepared.append((key, inputs))
        metadata.append(purpose_metadata(key, inputs, pins, procedure_hash, purpose))
    calibration = calibration_metadata(fixtures, procedure, pins, procedure_hash)
    formal_lookup = {c["fixture_id"]: c for c in fixtures["science"]}
    formal_metadata = [
        purpose_metadata(
            key,
            {
                **formal_lookup[key["fixture_id"]],
                "frequency_hz": key["frequency_hz"],
                "g_si": "6.6743e-11",
            },
            pins,
            procedure_hash,
        )
        for key in fixtures["expected_keys"]
    ]
    separation = common_v2.check_calibration_formal_separation(
        calibration + metadata if purpose == "calibration" else calibration,
        formal_metadata,
    )
    if not separation["admitted"]:
        raise ValueError("calibration/formal input or key overlap")
    # Persist predeclared purpose metadata before any physical evaluation.
    if metadata_path:
        with Path(metadata_path).open("xb") as output:
            output.write(b"".join(canonical_json_bytes(m) + b"\n" for m in metadata))
            output.flush()
            os.fsync(output.fileno())
    record_bindings = []
    diagnostics = []
    for key, inputs in prepared:
        inputs = dict(lookup[key["fixture_id"]])
        inputs["frequency_hz"] = key["frequency_hz"]
        inputs["g_si"] = "6.6743e-11"
        try:
            candidate = generator.reference(inputs, key["representation"])
            gate = acceptance.assess_numeric(
                candidate["quantity"],
                candidate["quantity"],
                candidate["source_scale"],
                "1e-10",
                "1e-9",
                candidate["uncertainty"],
            )
            if not gate["numeric_accepted"]:
                raise ValueError("frozen reference uncertainty/budget gate unresolved")
            status = "resolved"
        except Exception as error:
            candidate = {"quantity": None, "uncertainty": None}
            status = "unresolved"
            print(type(error).__name__ + ": " + str(error), file=sys.stderr, flush=True)
        record = {
            "schema_version": 1,
            "key": key,
            "status": status,
            "quantity": candidate["quantity"],
            "uncertainty": candidate["uncertainty"],
            "input_assumptions": "exact_input",
            "source_pins": pins,
            "procedure_hash": digest(procedure_path),
            "inputs": inputs,
            "units": "m^2 s^-4 Hz^-1"
            if key["representation"] == "SI"
            else "kg^2 m^-4 Hz^-1",
            "input_convention": procedure["input_convention"],
            "admission_status": "NOT_FORMALLY_ADMITTED",
        }
        acceptance.validate_record(record)
        record_bindings.append(
            {
                "key": key,
                "record_sha256": hashlib.sha256(
                    canonical_json_bytes(record) + b"\n"
                ).hexdigest(),
                "purpose_metadata_sha256": hashlib.sha256(
                    canonical_json_bytes(metadata[len(record_bindings)]) + b"\n"
                ).hexdigest(),
            }
        )
        diagnostics.append(
            {
                "key": key,
                "status": status,
                "source_scale": candidate.get("source_scale"),
                "method": candidate.get("method"),
            }
        )
        raw_record = canonical_json_bytes(record) + b"\n"
        if records_path is None:
            sys.stdout.buffer.write(raw_record)
            sys.stdout.buffer.flush()
        else:
            # The admitted supervisor seals this file, not stdout.log. Never
            # replace an existing stream; flush each row to preserve failures.
            with Path(records_path).open(
                "xb" if len(record_bindings) == 1 else "ab"
            ) as output:
                output.write(raw_record)
                output.flush()
                os.fsync(output.fileno())
    final = inventory()
    if final != observed:
        raise RuntimeError(
            "runtime import closure or source bytes changed during computation"
        )
    if environment_path:
        with Path(environment_path).open("xb") as output:
            output.write(canonical_json_bytes(final) + b"\n")
    if metadata_path:
        root = Path(metadata_path).parent
        for name, value in (
            ("record-bindings.json", record_bindings),
            ("reference-diagnostics.json", diagnostics),
            (
                "calibration-separation.json",
                {"calibration": calibration, "result": separation},
            ),
        ):
            with (root / name).open("xb") as output:
                output.write(canonical_json_bytes(value) + b"\n")


def validate_reviewed_sources(manifest, reviewed_commit):
    groups = manifest["reviewed_source_hashes_by_group"]
    expected_paths = {
        "generator": {
            "tests/oracles/gen1/stochastic_formal/" + f + "/generator.py"
            for f in ("r1", "b2")
        },
        "runner": {
            "tests/oracles/gen1/stochastic_formal/" + f + "/runner.py"
            for f in ("r1", "b2")
        },
        "sealer_helpers": {
            "tests/oracles/gen1/" + f
            for f in (
                "stochastic_acceptance.py",
                "stochastic_f_common_v2.py",
                "stochastic_attempt.py",
                "run_stochastic_local.py",
            )
        },
    }
    if set(groups) != set(expected_paths) or any(
        set(groups[g]) != paths for g, paths in expected_paths.items()
    ):
        raise ValueError(
            "exact joint-family three-group reviewed source closure required"
        )
    pinned = {
        p: h
        for group in manifest["file_hashes_by_group"].values()
        for p, h in group.items()
    }
    for group in groups.values():
        for relative, sha in group.items():
            if (
                pinned.get(relative) != sha
                or digest(Path(manifest["repository_root"]) / relative) != sha
            ):
                raise ValueError(
                    "reviewed source differs from E working bytes/admitted closure"
                )
            for commit in (reviewed_commit, manifest["E_commit"]):
                blob = subprocess.run(
                    [
                        "git",
                        "-C",
                        manifest["repository_root"],
                        "show",
                        commit + ":" + relative,
                    ],
                    capture_output=True,
                )
                if blob.returncode or hashlib.sha256(blob.stdout).hexdigest() != sha:
                    raise ValueError(
                        "reviewed source differs from candidate/E committed blob"
                    )


def verify_formal_manifest(args):
    if not args.manifest:
        raise ValueError("formal worker requires reviewed durable E manifest")
    manifest = parse_json(Path(args.manifest).read_bytes())
    review = parse_json(Path(manifest["review_receipt"]).read_bytes())
    if digest(manifest["review_receipt"]) != manifest["review_receipt_sha256"]:
        raise ValueError("review receipt byte mismatch")
    if (
        review["family"] != FAMILY
        or set(review["reviews"]) != {"science", "numerical", "provenance"}
        or any(v != "PASS" for v in review["reviews"].values())
    ):
        raise ValueError("three independent reviewed PASS required")
    if (
        review["file_hashes_by_group"] != manifest["file_hashes_by_group"]
        or review["reviewed_source_hashes_by_group"]
        != manifest["reviewed_source_hashes_by_group"]
    ):
        raise ValueError("review exact source pins mismatch")
    validate_durable_e(manifest)
    validate_freeze(manifest, args.fixtures, args.procedure)
    reviewer_ids = set()
    for role in ("science", "numerical", "provenance"):
        artifact = review["review_artifacts"][role]
        if digest(artifact["path"]) != artifact["sha256"]:
            raise ValueError("independent review artifact byte mismatch")
        actual = parse_json(Path(artifact["path"]).read_bytes())
        required = {
            "schema_version",
            "role",
            "reviewer_id",
            "decision",
            "scope",
            "reviewed_commit",
            "families",
            "file_hashes_by_group",
            "original_reviews",
        }
        if (
            set(actual) != required
            or actual["schema_version"] != 1
            or actual["role"] != role
            or actual["decision"] != "PASS"
            or actual["scope"] != "reference implementation"
            or not isinstance(actual["families"], list)
            or len(actual["families"]) != 2
            or set(actual["families"]) != {"R1", "B2"}
            or not isinstance(actual["reviewed_commit"], str)
            or len(actual["reviewed_commit"]) != 40
            or any(c not in "0123456789abcdef" for c in actual["reviewed_commit"])
            or actual["file_hashes_by_group"]
            != manifest["reviewed_source_hashes_by_group"]
            or not isinstance(actual["reviewer_id"], str)
            or not actual["reviewer_id"]
            or actual["reviewer_id"] in reviewer_ids
        ):
            raise ValueError(
                "implementation review semantic source/role binding mismatch"
            )
        validate_reviewed_sources(manifest, actual["reviewed_commit"])
        reviewer_ids.add(actual["reviewer_id"])
        ancestry = subprocess.run(
            [
                "git",
                "-C",
                manifest["repository_root"],
                "merge-base",
                "--is-ancestor",
                actual["reviewed_commit"],
                manifest["E_commit"],
            ],
            capture_output=True,
            text=True,
        )
        if ancestry.returncode:
            raise ValueError("reviewed source commit is not an ancestor of E")
        if (
            not isinstance(actual["original_reviews"], list)
            or not actual["original_reviews"]
        ):
            raise ValueError("original implementation reports required")
        for original in actual["original_reviews"]:
            if (
                set(original) != {"path", "sha256"}
                or digest(original["path"]) != original["sha256"]
            ):
                raise ValueError("original implementation report byte mismatch")
    procedure = parse_json(Path(args.procedure).read_bytes())
    source_artifacts = manifest["source_artifacts"]
    if len(source_artifacts) != len(procedure["source_pins"]):
        raise ValueError("complete primary source bytes required")
    for artifact, pin in zip(source_artifacts, procedure["source_pins"], strict=True):
        if (
            artifact["sha256"] != pin["sha256"]
            or digest(artifact["path"]) != pin["sha256"]
        ):
            raise ValueError("primary source artifact byte mismatch")
    # Frozen source/infrastructure pin verifier checks actual clean head/tree.
    fixtures = parse_json(Path(args.fixtures).read_bytes())
    for path in (args.fixtures, args.procedure, __file__, str(HERE / "generator.py")):
        rel = str(
            Path(path)
            .resolve()
            .relative_to(Path(manifest["repository_root"]).resolve())
        )
        pinned = {
            p: h
            for group in manifest["file_hashes_by_group"].values()
            for p, h in group.items()
        }
        if pinned.get(rel) != digest(path):
            raise ValueError("runner argument not in exact reviewed closure")
    required_helpers = {
        "tests/oracles/gen1/stochastic_f_common_v2.py",
        "tests/oracles/gen1/stochastic_acceptance.py",
        "tests/oracles/gen1/stochastic_attempt.py",
        "tests/oracles/gen1/run_stochastic_local.py",
    }
    baseline = manifest["common_baseline_hashes"]
    if set(baseline) != set(common_v2.NORMATIVE_COMMON_PATHS):
        raise ValueError("exact seven-file common integrity boundary required")
    for relative, sha in baseline.items():
        if (
            pinned.get(relative) != sha
            or digest(Path(manifest["repository_root"]) / relative) != sha
        ):
            raise ValueError("frozen common source bytes changed or unpinned")
    if not required_helpers.issubset(pinned):
        raise ValueError(
            "review closure omits actual common computational/transport imports"
        )
    verify_source_pin(
        manifest["repository_root"],
        expected_head=manifest["E_commit"],
        expected_tree=manifest["E_tree"],
        file_hashes_by_group=manifest["file_hashes_by_group"],
    )
    verify_environment_inventory(
        manifest["environment_inventory_path"], manifest["environment_inventory_sha256"]
    )
    return manifest, fixtures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument(
        "--purpose", choices=["calibration", "formal-science"], default="formal-science"
    )
    parser.add_argument("--expected-environment")
    parser.add_argument("--metadata")
    parser.add_argument("--records", default="records.raw.jsonl")
    parser.add_argument("--environment")
    parser.add_argument("--fixtures", required=True)
    parser.add_argument("--procedure", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--attempt")
    args = parser.parse_args()
    if args.inventory:
        print(json.dumps(inventory(), sort_keys=True, separators=(",", ":")))
        return
    if args.worker:
        if args.purpose == "formal-science":
            if Path(args.records).resolve() != Path.cwd() / "records.raw.jsonl":
                raise ValueError("formal worker must write admitted records.raw.jsonl")
            manifest, _ = verify_formal_manifest(args)
            if (
                not args.expected_environment
                or Path(args.expected_environment).resolve()
                != Path(manifest["environment_inventory_path"]).resolve()
            ):
                raise ValueError("formal worker requires manifest-pinned environment")
        emit(
            args.fixtures,
            args.procedure,
            args.metadata,
            args.environment,
            args.expected_environment,
            args.purpose,
            args.records,
        )
        return
    if not args.manifest or not args.attempt:
        parser.error("formal requires reviewed durableE manifest and freshattempt")
    validate_external_attempt(args.attempt)
    manifest, fixtures = verify_formal_manifest(args)
    command = [
        sys.executable,
        "-I",
        str(Path(__file__).resolve()),
        "--worker",
        "--manifest",
        str(Path(args.manifest).resolve()),
        "--fixtures",
        str(Path(args.fixtures).resolve()),
        "--procedure",
        str(Path(args.procedure).resolve()),
        "--expected-environment",
        str(Path(manifest["environment_inventory_path"]).resolve()),
        "--records",
        str(Path(args.attempt).resolve() / "records.raw.jsonl"),
        "--metadata",
        str(Path(args.attempt).resolve() / "reference-metadata.jsonl"),
        "--environment",
        str(Path(args.attempt).resolve() / "worker-environment.json"),
    ]
    seal = run_frozen_attempt(
        args.attempt,
        repository_root=manifest["repository_root"],
        command=command,
        key_fields=KEY_FIELDS,
        expected_keys=fixtures["expected_keys"],
        record_key_path=["key"],
        expected_head=manifest["E_commit"],
        expected_tree=manifest["E_tree"],
        file_hashes_by_group=manifest["file_hashes_by_group"],
        environment_inventory_path=manifest["environment_inventory_path"],
        environment_inventory_sha256=manifest["environment_inventory_sha256"],
        timeout_seconds=1800,
        environment={**os.environ, "PYTHONPATH": ""},
    )
    print(
        json.dumps(
            {
                "seal_sha256": digest(Path(args.attempt) / "seal.json"),
                "execution": seal["execution"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
