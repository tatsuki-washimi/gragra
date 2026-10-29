"""Authorized B1 production comparison driver.

Importing this module, requesting ``--help`` or running preflight does not
import the production package or evaluate any production function. Execution
requires a separately reviewed authorization binding and writes durable
partial evidence before loading/calling the production API.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable

from tests.oracles.gen1.b1_exact_record_adapter import (
    KEY_FIELDS,
    adapt_b1_records,
    fraction_to_exact_text,
    key_tuple,
    parse_exact_fraction,
    strict_json_loads,
)


PRODUCTION_COMMIT = "31e1a0e3cb9bca32cc7e00722580d73fcd2241d0"
PRODUCTION_TREE = "e47d7b6a4bbb110b755138234f656b4bad248bbb"
KERNEL_PATH = "tests/oracles/gen1/stochastic_acceptance.py"
KERNEL_GIT_BLOB = "7be0818890d55350ed5bf538f676f903c3709a5e"
KERNEL_SHA256 = "09dbe97ceaf635b0fd0eafdf2cb3f11ac994af276e318a684c45c00a5548be68"
KERNEL_COMMIT = PRODUCTION_COMMIT
REFERENCE_SHA256 = {
    "review": "c8dc5adf0aaf04d20598eb1f8a11b038990f16f28a1c75f6b82d331a77398c08",
    "records": "0938f679cc68e6a39cfd5e3f140c5fbc82b06969a4cae275367a0bc588064bb0",
    "seal": "44fbb36c9eb3a5c8d9f1a1a1bd12966a6a3487fee547c2371a3b70e37bec901e",
}
FROZEN_SOURCE_PATHS = {
    "docs/design/gen1/stochastic/B1-budget.json",
    "docs/design/gen1/stochastic/B1-fixtures.json",
    "docs/design/gen1/stochastic/B1-freeze-v2.json",
    "docs/design/gen1/stochastic/B1-procedure-v2.json",
    "docs/design/gen1/stochastic/B1-procedure.json",
    "tests/oracles/gen1/b1_analytic_enclosure.py",
    "tests/oracles/gen1/b1_formal_generator.py",
    "tests/oracles/gen1/run_stochastic_seismic_reference.py",
    "tests/oracles/gen1/run_stochastic_local.py",
    "tests/oracles/gen1/stochastic_attempt.py",
    "tests/test_gen1_stochastic_pipeline.py",
    "docs/design/gen1/stochastic/common-freeze-v2.json",
    "docs/design/gen1/stochastic/common-policy.json",
    "tests/oracles/gen1/TRACE-F_B1_v2-obligations.md",
}
_EXPECTED_API_PARAMETERS = {
    "gfree": {
        "density_kg_m3": "KEYWORD_ONLY",
        "frequency_hz": "KEYWORD_ONLY",
        "p_displacement_psd_m2_hz": "KEYWORD_ONLY",
    },
    "si": {
        "density_kg_m3": "KEYWORD_ONLY",
        "frequency_hz": "KEYWORD_ONLY",
        "p_displacement_psd_m2_hz": "KEYWORD_ONLY",
    },
}
CONTRACT_GATES = (
    "units",
    "shape",
    "frequency_axis",
    "input_normalization",
    "source_connections",
    "resolved_reference",
    "sealed_exact_coverage",
)
B1_AUTHORIZATION_CHECKS = {
    "accepted_reference",
    "adapter_exact",
    "all_30_keys_exact",
    "b1_001_preserved",
    "comparator_commit",
    "comparator_policy",
    "comparator_sources",
    "environment",
    "frozen_sources",
    "full_csd_comparator",
    "production_execution_not_run",
    "production_pin",
    "source_connections_70",
    "strict_zero_12",
}
AUTHORIZATION_ARTIFACTS = {
    "candidate",
    "verifier",
    "verification_output",
    "implementation_review",
    "authorization_review",
}
_BINDING_FIELDS = {
    "schema_version",
    "authorization_status",
    "production_comparison_run",
    "artifacts",
    "pins",
}
_CANDIDATE_FIELDS = {"schema_version", "artifact_id", "authorization", "pins", "checks"}
_CANDIDATE_AUTH_FIELDS = {
    "status",
    "authorization_granted",
    "production_comparison_run",
    "production_acceptance_established",
}
_IMPLEMENTATION_REVIEW_FIELDS = {
    "schema_version",
    "decision",
    "reviewed_commit",
    "reviewed_tree",
    "blocking_findings",
    "reviewed_files",
    "test_command",
    "test_exit_code",
    "test_output_sha256",
    "production_execution",
    "pins",
}
_AUTHORIZATION_REVIEW_FIELDS = {
    "schema_version",
    "decision",
    "candidate_sha256",
    "implementation_review_sha256",
    "verification_output_sha256",
    "pins",
    "production_comparison_run",
    "blockers",
}
_VERIFICATION_FIELDS = {
    "schema_version",
    "artifact_id",
    "authorization",
    "gate_review",
    "candidate_sha256",
    "verifier_sha256",
    "production_comparison_run",
    "checks",
    "counts",
}
_POLICY_FIELDS = {
    "schema_version",
    "production_repo",
    "production_commit",
    "production_tree",
    "production_git_blobs",
    "frozen_source_sha256",
    "reference_paths",
    "reference_sha256",
    "fixture_path",
    "budget_path",
    "procedure_path",
    "environment",
    "production_modules",
    "production_api_parameters",
    "g_si",
    "exact_zero_fixture_ids",
    "source_connection_receipt_path",
    "source_connection_receipt_sha256",
    "authorization_pin_contract",
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class PreflightError(RuntimeError):
    """A required authorization, provenance, or runtime preflight failed."""


class AuthorizationError(PreflightError):
    """The external authorization binding is absent, stale, or invalid."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    _reject_symlink_path(path)
    if not path.is_file():
        raise PreflightError(f"expected regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_symlink_path(path: Path) -> None:
    path = path.absolute()
    for item in (path, *path.parents):
        try:
            if item.is_symlink():
                raise PreflightError(f"symlink path is forbidden: {item}")
        except OSError as exc:
            raise PreflightError(f"cannot inspect path: {item}") from exc


def _read_json(path: str | Path) -> Any:
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise PreflightError(f"cannot read JSON artifact: {path}") from exc
    try:
        return strict_json_loads(raw)
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise PreflightError(f"invalid strict JSON artifact: {path}") from exc


def verify_pin_map(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    """Require exact key/value equality, including no unknown pins."""
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        raise PreflightError("pin maps must be objects")
    if set(expected) != set(actual):
        missing = sorted(set(expected) - set(actual))
        unknown = sorted(set(actual) - set(expected))
        raise PreflightError(f"pin map key mismatch; missing={missing}, unknown={unknown}")
    if expected != actual:
        raise PreflightError("pinned values do not match current values")
    return True


def verify_authorization_pin_contract(expected_static: dict[str, Any], actual: dict[str, Any]) -> bool:
    """Check policy pins plus the two review-derived self revision pins."""
    if not isinstance(expected_static, dict) or not isinstance(actual, dict):
        raise PreflightError("authorization pin maps must be objects")
    dynamic = {"comparator_commit", "comparator_tree"}
    expected_fields = set(expected_static)
    if dynamic <= expected_fields:
        return verify_pin_map(expected_static, actual)
    if set(actual) != expected_fields | dynamic:
        raise PreflightError("authorization pin map key mismatch")
    if any(not isinstance(actual[name], str) or not actual[name] for name in dynamic):
        raise PreflightError("review-derived comparator revision pin is missing")
    for name, value in expected_static.items():
        if actual[name] != value:
            raise PreflightError("authorization static pin mismatch")
    return True


def _resolve_artifact_path(binding_path: Path, artifact_path: Any) -> Path:
    if not isinstance(artifact_path, str) or not artifact_path:
        raise AuthorizationError("authorization artifact path is missing")
    path = Path(artifact_path)
    if not path.is_absolute():
        path = binding_path.parent / path
    _reject_symlink_path(path)
    return path


def verify_authorization_bundle(
    authorization_path: str | Path,
    expected_authorization_sha256: str,
    expected_review_sha256: str,
    expected_pins: dict[str, Any],
) -> dict[str, Any]:
    """Verify the external trust binding and every reviewed artifact digest.

    ``expected_authorization_sha256`` and ``expected_review_sha256`` are
    operator-supplied trust anchors, not values read from the binding itself.
    """
    binding_path = Path(authorization_path)
    try:
        actual_binding_sha256 = sha256_file(binding_path)
    except PreflightError as exc:
        raise AuthorizationError("authorization binding is unreadable or missing") from exc
    if not _SHA256.fullmatch(expected_authorization_sha256) or actual_binding_sha256 != expected_authorization_sha256:
        raise AuthorizationError("authorization binding hash mismatch")
    if not _SHA256.fullmatch(expected_review_sha256):
        raise AuthorizationError("expected authorization review digest is malformed")
    binding = _read_json(binding_path)
    if not isinstance(binding, dict) or set(binding) != _BINDING_FIELDS:
        raise AuthorizationError("authorization binding schema mismatch")
    if type(binding["schema_version"]) is not int or binding["schema_version"] != 1:
        raise AuthorizationError("authorization binding version mismatch")
    if binding["authorization_status"] != "AUTHORIZED":
        raise AuthorizationError("P_science is not AUTHORIZED")
    if binding["production_comparison_run"] is not False:
        raise AuthorizationError("authorization binding claims comparison already ran")
    if not isinstance(binding["artifacts"], dict) or set(binding["artifacts"]) != AUTHORIZATION_ARTIFACTS:
        raise AuthorizationError("authorization artifact inventory mismatch")
    try:
        verify_authorization_pin_contract(expected_pins, binding["pins"])
    except PreflightError as exc:
        raise AuthorizationError("authorization pin map mismatch") from exc

    artifact_bytes: dict[str, bytes] = {}
    artifact_values: dict[str, Any] = {}
    for name, artifact in binding["artifacts"].items():
        if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256"}:
            raise AuthorizationError(f"invalid {name} artifact binding")
        digest = artifact["sha256"]
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise AuthorizationError(f"invalid {name} artifact digest")
        try:
            path = _resolve_artifact_path(binding_path, artifact["path"])
            actual_digest = sha256_file(path)
        except PreflightError as exc:
            raise AuthorizationError(f"{name} artifact is missing or unreadable") from exc
        if actual_digest != digest:
            raise AuthorizationError(f"{name} artifact hash mismatch")
        data = path.read_bytes()
        artifact_bytes[name] = data
        if name != "verifier":
            artifact_values[name] = strict_json_loads(data)

    candidate = artifact_values["candidate"]
    verification = artifact_values["verification_output"]
    implementation = artifact_values["implementation_review"]
    review = artifact_values["authorization_review"]
    if not isinstance(candidate, dict) or set(candidate) != _CANDIDATE_FIELDS:
        raise AuthorizationError("candidate schema mismatch")
    if type(candidate["schema_version"]) is not int or candidate["schema_version"] != 1:
        raise AuthorizationError("candidate version mismatch")
    if not isinstance(candidate["artifact_id"], str) or not candidate["artifact_id"]:
        raise AuthorizationError("candidate artifact ID is missing")
    if not isinstance(candidate["authorization"], dict) or set(candidate["authorization"]) != _CANDIDATE_AUTH_FIELDS:
        raise AuthorizationError("candidate authorization schema mismatch")
    candidate_decision = candidate["authorization"]
    if (
        candidate_decision.get("status") != "AUTHORIZED"
        or candidate_decision.get("authorization_granted") is not True
        or candidate_decision.get("production_comparison_run") is not False
        or candidate_decision.get("production_acceptance_established") is not False
    ):
        raise AuthorizationError("candidate decision is not a pre-comparison authorization")
    if "pins" not in candidate:
        raise AuthorizationError("candidate does not carry exact pins")
    if candidate["pins"] != binding["pins"]:
        raise AuthorizationError("candidate pin map differs from binding")

    if not isinstance(candidate["checks"], dict) or set(candidate["checks"]) != B1_AUTHORIZATION_CHECKS:
        raise AuthorizationError("candidate gate inventory is incomplete or has unknown gates")
    if any(status != "PASS" for status in candidate["checks"].values()):
        raise AuthorizationError("authorized candidate contains a non-PASS gate")

    if not isinstance(verification, dict) or set(verification) != _VERIFICATION_FIELDS:
        raise AuthorizationError("verification output is malformed")
    if type(verification["schema_version"]) is not int or verification["schema_version"] != 1:
        raise AuthorizationError("verification output version mismatch")
    if verification["artifact_id"] != candidate["artifact_id"]:
        raise AuthorizationError("verification output artifact ID mismatch")
    if (
        verification.get("authorization") != "AUTHORIZED"
        or verification.get("gate_review") != "PASS"
        or verification.get("candidate_sha256") != binding["artifacts"]["candidate"]["sha256"]
        or verification.get("verifier_sha256") != binding["artifacts"]["verifier"]["sha256"]
        or verification.get("production_comparison_run") is not False
    ):
        raise AuthorizationError("authorization verification output does not pass")
    if not isinstance(verification["checks"], dict) or verification["checks"] != candidate["checks"]:
        raise AuthorizationError("verification output does not reproduce candidate gate inventory")
    if not isinstance(verification["counts"], dict) or set(verification["counts"]) != {"PASS", "BLOCKER", "FAIL"}:
        raise AuthorizationError("verification counts schema mismatch")
    expected_counts = {
        "PASS": len(candidate["checks"]),
        "BLOCKER": 0,
        "FAIL": 0,
    }
    if verification["counts"] != expected_counts:
        raise AuthorizationError("verification counts do not match the gate inventory")
    if any(type(value) is not int for value in verification["counts"].values()):
        raise AuthorizationError("verification counts must be exact integers")

    if not isinstance(implementation, dict) or set(implementation) != _IMPLEMENTATION_REVIEW_FIELDS:
        raise AuthorizationError("independent implementation review schema mismatch")
    if type(implementation["schema_version"]) is not int or implementation["schema_version"] != 1:
        raise AuthorizationError("implementation review version mismatch")
    if (
        implementation.get("decision") != "APPROVED"
        or type(implementation.get("blocking_findings")) is not int
        or implementation["blocking_findings"] != 0
        or implementation.get("reviewed_commit") != binding["pins"].get("comparator_commit")
        or implementation.get("reviewed_tree") != binding["pins"].get("comparator_tree")
        or implementation.get("pins") != binding["pins"]
        or not isinstance(implementation.get("reviewed_files"), list)
        or any(not isinstance(item, str) for item in implementation["reviewed_files"])
        or set(implementation["reviewed_files"]) != {
            "tests/oracles/gen1/compare_stochastic_production.py",
            "tests/oracles/gen1/b1_exact_record_adapter.py",
            "tests/oracles/gen1/b1-production-comparator-policy-v1.json",
            "tests/oracles/gen1/README.md",
            "tests/test_gen1_b1_exact_record_adapter.py",
            "tests/test_gen1_b1_production_comparator.py",
        }
        or len(implementation["reviewed_files"]) != len(set(implementation["reviewed_files"]))
        or not isinstance(implementation.get("test_command"), str)
        or type(implementation.get("test_exit_code")) is not int
        or implementation["test_exit_code"] != 0
        or not isinstance(implementation.get("test_output_sha256"), str)
        or not _SHA256.fullmatch(implementation["test_output_sha256"])
        or implementation.get("production_execution") is not False
    ):
        raise AuthorizationError("independent implementation review is not approved for pinned code")

    review_digest = binding["artifacts"]["authorization_review"]["sha256"]
    if review_digest != expected_review_sha256:
        raise AuthorizationError("authorization review digest does not match operator pin")
    if not isinstance(review, dict) or set(review) != _AUTHORIZATION_REVIEW_FIELDS:
        raise AuthorizationError("authorization review schema mismatch")
    if type(review["schema_version"]) is not int or review["schema_version"] != 1:
        raise AuthorizationError("authorization review version mismatch")
    if (
        review.get("decision") != "AUTHORIZED"
        or review.get("candidate_sha256") != binding["artifacts"]["candidate"]["sha256"]
        or review.get("implementation_review_sha256")
        != binding["artifacts"]["implementation_review"]["sha256"]
        or review.get("verification_output_sha256")
        != binding["artifacts"]["verification_output"]["sha256"]
        or review.get("production_comparison_run") is not False
        or type(review.get("blockers")) is not int
        or review["blockers"] != 0
    ):
        raise AuthorizationError("authorization review does not bind the candidate and implementation review")
    if review["pins"] != binding["pins"]:
        raise AuthorizationError("authorization review pin map mismatch")
    return binding


def validate_output_parent(path: str | Path, repository_roots: Iterable[str | Path]) -> Path:
    """Validate a nonexistent output directory outside every repository."""
    path = Path(path).absolute()
    _reject_symlink_path(path)
    if path.exists():
        raise PreflightError(f"comparison output must be fresh: {path}")
    resolved = path.resolve(strict=False)
    for root in repository_roots:
        repository = Path(root).resolve(strict=False)
        if resolved == repository or resolved.is_relative_to(repository) or repository.is_relative_to(resolved):
            raise PreflightError(f"comparison output overlaps a repository: {resolved}")
    return resolved


def load_pinned_kernel(
    production_repo: str | Path,
    *,
    commit: str = KERNEL_COMMIT,
    path: str = KERNEL_PATH,
    expected_git_blob: str = KERNEL_GIT_BLOB,
    expected_sha256: str = KERNEL_SHA256,
) -> ModuleType:
    """Load the exact pure arithmetic helper from a verified Git object."""
    repository = Path(production_repo)
    try:
        blob = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", f"{commit}:{path}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        source = subprocess.run(
            ["git", "-C", str(repository), "show", f"{commit}:{path}"],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PreflightError("cannot read pinned arithmetic-kernel Git object") from exc
    if blob != expected_git_blob or _sha256_bytes(source) != expected_sha256:
        raise PreflightError("arithmetic-kernel blob or SHA-256 mismatch")
    module = ModuleType("_pinned_b1_stochastic_acceptance")
    module.__file__ = f"{commit}:{path}"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    for symbol in ("promote_binary64", "assess", "structure_gates", "canonical_json"):
        if not callable(getattr(module, symbol, None)):
            raise PreflightError(f"pinned arithmetic kernel lacks {symbol}")
    return module


def matrix_to_quantity(matrix: Any, kernel: ModuleType | Any | None = None) -> dict[str, Any]:
    """Promote a real finite float64 3×3 production result exactly."""
    import numpy as np

    if not isinstance(matrix, np.ndarray):
        raise ValueError("production CSD must be a NumPy array")
    if matrix.shape != (3, 3):
        raise ValueError("production CSD shape must be exactly (3,3)")
    if matrix.dtype != np.dtype(np.float64):
        raise ValueError("production CSD dtype must be real float64")
    if matrix.flags.writeable:
        raise ValueError("production CSD must be read-only")
    if not np.isfinite(matrix).all():
        raise ValueError("production CSD contains NaN or infinity")
    kernel = kernel or load_pinned_kernel(
        Path(__file__).resolve().parents[4] / "stoch-integration-20261002"
    )
    values = []
    for value in matrix.flat:
        promoted = kernel.promote_binary64(float(value))
        values.append({"real": promoted, "imag": "0e+0"})
    return {"shape": [3, 3], "data": values}


def build_production_inputs(key: dict[str, Any], fixture: dict[str, Any]) -> dict[str, Any]:
    """Map exact fixture inputs to the scalar public API boundary."""
    if key.get("fixture_id") != fixture.get("fixture_id"):
        raise PreflightError("key-to-fixture mapping mismatch")
    if (
        key.get("observable") != "acceleration_csd"
        or key.get("target_id") != fixture.get("target_id")
        or fixture.get("frame") != "Cartesian isotropic natural frame"
    ):
        raise PreflightError("frozen target, observable, or frame mapping mismatch")
    target = fixture.get("target_m")
    if not isinstance(target, list) or len(target) != 3 or any(parse_exact_fraction(x) != 0 for x in target):
        raise PreflightError("frozen origin target is not exact zero")
    if fixture.get("input_assumptions") != "exact_input":
        raise PreflightError("B1 fixture input assumption is not exact_input")
    frequencies = fixture.get("frequency_hz")
    psd_values = fixture.get("displacement_psd_m2_hz")
    if not isinstance(frequencies, list) or not isinstance(psd_values, list) or len(frequencies) != len(psd_values):
        raise PreflightError("fixture frequency/PSD axes are malformed")
    matches = [index for index, value in enumerate(frequencies) if value == key.get("frequency_hz")]
    if len(matches) != 1:
        raise PreflightError("frequency key is missing or duplicated on the fixture axis")
    frequency_index = matches[0]
    density = parse_exact_fraction(fixture["density_kg_m3"])
    frequency = parse_exact_fraction(frequencies[frequency_index])
    psd = parse_exact_fraction(psd_values[frequency_index])
    convention = fixture.get("input_convention")
    if convention == "total":
        psd /= 3
    elif convention != "component":
        raise PreflightError("unknown B1 component/total input convention")
    if min(density, frequency, psd) < 0 or frequency == 0:
        raise PreflightError("fixture physical input is negative or frequency is zero")

    exact = {
        "density_kg_m3": density,
        "frequency_hz": frequency,
        "p_displacement_psd_m2_hz": psd,
    }
    floating: dict[str, float] = {}
    for name, value in exact.items():
        boundary_value = float(value)
        if not math.isfinite(boundary_value):
            raise PreflightError(f"fixture input overflows binary64: {name}")
        floating[name] = boundary_value
    evidence = {
        "exact_inputs": {name: fraction_to_exact_text(value) for name, value in exact.items()},
        "binary64_hex": {name: value.hex() for name, value in floating.items()},
        "input_convention": convention,
        "psd_total_divided_by_three": convention == "total",
    }
    return {"kwargs": floating, "evidence": evidence, "frequency_index": frequency_index}


def validate_source_connection_receipt(
    receipt: dict[str, Any],
    fixture_manifest: dict[str, Any],
    *,
    production_commit: str,
    production_tree: str,
    fixture_sha256: str,
    gate_fields: list[str],
    expected_science_keys: list[dict[str, Any]],
) -> dict[tuple[str, ...], bool]:
    """Validate the separately sealed 56+14 auxiliary connection evidence.

    A successful receipt is the only route to an explicit science-key
    ``source_connections=True`` contract gate. Candidate or policy booleans do
    not substitute for the per-key auxiliary evidence.
    """
    if set(receipt) != {
        "schema_version",
        "decision",
        "production_commit",
        "production_tree",
        "fixture_sha256",
        "results",
    }:
        raise PreflightError("source-connection receipt schema mismatch")
    if type(receipt["schema_version"]) is not int or receipt["schema_version"] != 1:
        raise PreflightError("source-connection receipt version mismatch")
    if (
        receipt["decision"] != "PASS"
        or receipt["production_commit"] != production_commit
        or receipt["production_tree"] != production_tree
        or receipt["fixture_sha256"] != fixture_sha256
    ):
        raise PreflightError("source-connection receipt pin or decision mismatch")
    if not isinstance(gate_fields, list) or not gate_fields or len(gate_fields) != len(set(gate_fields)):
        raise PreflightError("source-connection gate field manifest is malformed")

    ref_keys = fixture_manifest.get("expected_reference_auxiliary_keys")
    consistency_keys = fixture_manifest.get("expected_consistency_auxiliary_keys")
    all_keys = fixture_manifest.get("expected_auxiliary_keys")
    if not all(isinstance(group, list) for group in (ref_keys, consistency_keys, all_keys)):
        raise PreflightError("frozen auxiliary key partitions are missing")

    def auxiliary_id(key: Any) -> tuple[str, ...]:
        if not isinstance(key, dict) or set(key) != set(KEY_FIELDS):
            raise PreflightError("malformed auxiliary key")
        if any(not isinstance(key[field], str) or not key[field] for field in KEY_FIELDS):
            raise PreflightError("empty auxiliary key field")
        if key["set"] != "auxiliary" or key["representation"] not in {"Gfree", "SI"}:
            raise PreflightError("auxiliary key set/representation mismatch")
        if parse_exact_fraction(key["frequency_hz"]) <= 0:
            raise PreflightError("auxiliary key frequency is not positive")
        return tuple(key[field] for field in KEY_FIELDS)

    ref_sequence = [auxiliary_id(key) for key in ref_keys]
    consistency_sequence = [auxiliary_id(key) for key in consistency_keys]
    all_sequence = [auxiliary_id(key) for key in all_keys]
    if (
        len(ref_sequence) != 56
        or len(consistency_sequence) != 14
        or len(all_sequence) != 70
        or len(set(ref_sequence)) != len(ref_sequence)
        or len(set(consistency_sequence)) != len(consistency_sequence)
        or set(ref_sequence) & set(consistency_sequence)
        or set(all_sequence) != set(ref_sequence) | set(consistency_sequence)
    ):
        raise PreflightError("frozen 56+14 auxiliary partition is not exact")

    rows = receipt["results"]
    if not isinstance(rows, list) or len(rows) != 70:
        raise PreflightError("source-connection receipt must resolve all 70 auxiliary keys")
    expected_by_id = {
        **{key: "independent_reference" for key in ref_sequence},
        **{key: "production_consistency" for key in consistency_sequence},
    }
    observed: set[tuple[str, ...]] = set()
    row_fields = {
        "key",
        "partition",
        "status",
        "gate_results",
        "input_sha256",
        "result_sha256",
        "independent_reference_sha256",
    }
    for row in rows:
        if not isinstance(row, dict) or set(row) != row_fields:
            raise PreflightError("source-connection result schema mismatch")
        key_id = auxiliary_id(row["key"])
        if key_id in observed or key_id not in expected_by_id:
            raise PreflightError("duplicate or unknown auxiliary receipt key")
        observed.add(key_id)
        if row["partition"] != expected_by_id[key_id] or row["status"] != "resolved":
            raise PreflightError("source-connection key partition or status mismatch")
        gates = row["gate_results"]
        if not isinstance(gates, dict) or set(gates) != set(gate_fields):
            raise PreflightError("source-connection gate inventory mismatch")
        if any(value is not True for value in gates.values()):
            raise PreflightError("source-connection receipt has a non-PASS gate")
        for field in ("input_sha256", "result_sha256"):
            if not isinstance(row[field], str) or not _SHA256.fullmatch(row[field]):
                raise PreflightError(f"source-connection {field} is malformed")
        ref_digest = row["independent_reference_sha256"]
        if expected_by_id[key_id] == "independent_reference":
            if not isinstance(ref_digest, str) or not _SHA256.fullmatch(ref_digest):
                raise PreflightError("independent-reference receipt digest is missing")
        elif ref_digest is not None:
            raise PreflightError("consistency-only receipt claims an independent reference")
    if observed != set(all_sequence):
        raise PreflightError("source-connection receipt is missing auxiliary keys")
    science_ids = [key_tuple(key) for key in expected_science_keys]
    if len(science_ids) != 30 or len(set(science_ids)) != 30:
        raise PreflightError("source-connection receipt requires exact 30-key science mapping")
    return {key_id: True for key_id in science_ids}


def assess_one(
    *,
    key: dict[str, Any],
    fixture: dict[str, Any],
    reference: dict[str, Any],
    production_matrix: Any,
    expected_units: str,
    source_connections_passed: bool,
    exact_coverage_passed: bool,
    budget: dict[str, Any],
    kernel: ModuleType | Any,
) -> dict[str, Any]:
    """Apply full-matrix numeric, CSD-structure, and contract decisions."""
    if type(source_connections_passed) is not bool or type(exact_coverage_passed) is not bool:
        raise PreflightError("source-connection and coverage gates require explicit booleans")
    if reference.get("key") != key:
        raise PreflightError("reference key differs from requested science key")
    production = matrix_to_quantity(production_matrix, kernel)
    if reference.get("kernel_quantity", {}).get("shape") != [3, 3]:
        raise PreflightError("reference quantity is not a full CSD matrix")

    source_scale = reference["source_scale"]
    structural_rate = parse_exact_fraction(budget["structural_rate"])
    structural_tolerance = structural_rate * source_scale
    tolerance_text = fraction_to_exact_text(structural_tolerance)
    if "/" in tolerance_text:
        raise PreflightError("structural tolerance is not a terminating decimal")
    structure_conditions = kernel.structure_gates(
        production,
        symmetry="identity",
        tolerance=tolerance_text,
    )
    frequency_axis = fixture.get("frequency_hz")
    frequency_gate = (
        isinstance(frequency_axis, list)
        and frequency_axis.count(key.get("frequency_hz")) == 1
    )
    input_convention = fixture.get("input_convention")
    normalization = (
        fixture.get("input_assumptions") == "exact_input"
        and input_convention in {"component", "total"}
        and isinstance(fixture.get("displacement_psd_m2_hz"), list)
        and len(fixture["displacement_psd_m2_hz"]) == len(frequency_axis or [])
    )
    contract_conditions = {
        "units": reference.get("units") == expected_units,
        "shape": production["shape"] == [3, 3],
        "frequency_axis": frequency_gate,
        "input_normalization": normalization,
        "source_connections": source_connections_passed,
        "resolved_reference": reference.get("classification") in {"rigorous_bound", "exact_zero"},
        "sealed_exact_coverage": exact_coverage_passed,
    }
    assessment = kernel.assess(
        production=production,
        reference=reference["kernel_quantity"],
        source_scale=fraction_to_exact_text(source_scale),
        absolute_rate=budget["absolute_rate"],
        relative_rate=budget["relative_rate"],
        uncertainty=reference["kernel_uncertainty"],
        structure_conditions=structure_conditions,
        contract_conditions=contract_conditions,
    )
    is_zero = reference.get("is_exact_zero") is True
    zero_observed = all(
        item[part] == "0e+0"
        for item in production["data"]
        for part in ("real", "imag")
    )
    exact_zero_pass = (not is_zero) or (
        zero_observed
        and reference.get("u_matrix_exact") == 0
        and reference.get("u_matrix_kernel") == 0
        and source_scale == 0
        and assessment.get("difference_upper") == "0e+0"
    )
    accepted = assessment.get("accepted") is True and exact_zero_pass
    return {
        "key": key,
        "production_quantity": production,
        "structure_conditions": structure_conditions,
        "contract_conditions": contract_conditions,
        "assessment": assessment,
        "u_matrix_exact": fraction_to_exact_text(reference["u_matrix_exact"]),
        "u_matrix_kernel": fraction_to_exact_text(reference["u_matrix_kernel"]),
        "exact_zero_pass": exact_zero_pass,
        "accepted": accepted,
    }


def _assert_json_safe(value: Any) -> None:
    if isinstance(value, (float, Fraction)):
        raise ValueError("comparison artifacts may not contain binary floats/Fractions")
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("comparison artifact object keys must be strings")
        for item in value.values():
            _assert_json_safe(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_json_safe(item)


def _canonical_json_bytes(value: Any) -> bytes:
    _assert_json_safe(value)
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_new_fsynced(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise


def _replace_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_name(f".manifest-{uuid.uuid4().hex}.tmp")
    _write_new_fsynced(temporary, _canonical_json_bytes(manifest))
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _append_fsynced(path: Path, value: dict[str, Any]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "ab") as stream:
            stream.write(_canonical_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        _fsync_directory(path.parent)


def _format_exception(exc: BaseException) -> dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc).replace("\x00", "")[:2000],
    }


def execute_authorized_comparison(
    *,
    authorization_path: str | Path,
    expected_authorization_sha256: str,
    expected_review_sha256: str,
    expected_pins: dict[str, Any],
    expected_keys: list[dict[str, Any]],
    exact_zero_keys: list[dict[str, Any]],
    fixtures_by_id: dict[str, dict[str, Any]],
    references: dict[tuple[str, ...], dict[str, Any]],
    evaluator: Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], Any] | None = None,
    evaluator_factory: Callable[[], Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], Any]] | None = None,
    kernel: ModuleType | Any,
    budget: dict[str, Any],
    source_connection_results: dict[tuple[str, ...], bool],
    output_parent: str | Path,
    repository_roots: Iterable[str | Path],
) -> Path:
    """Run only after independently reviewed authorization and pin preflight."""
    binding = verify_authorization_bundle(
        authorization_path,
        expected_authorization_sha256,
        expected_review_sha256,
        expected_pins,
    )
    if (evaluator is None) == (evaluator_factory is None):
        raise PreflightError("provide exactly one evaluator or lazy evaluator factory")

    expected_ids = [key_tuple(key) for key in expected_keys]
    if len(expected_ids) != 30 or len(set(expected_ids)) != 30:
        raise PreflightError("B1 comparison requires the exact unique 30-key science set")
    if set(references) != set(expected_ids):
        raise PreflightError("reference mapping has missing or unknown science keys")
    if set(source_connection_results) != set(expected_ids) or any(
        type(value) is not bool for value in source_connection_results.values()
    ):
        raise PreflightError("source-connection gate mapping is incomplete or malformed")
    exact_zero_ids = [key_tuple(key) for key in exact_zero_keys]
    exact_zero_set = set(exact_zero_ids)
    if len(exact_zero_ids) != len(set(exact_zero_ids)) or set(exact_zero_ids) != {
        key for key in expected_ids if references[key].get("is_exact_zero") is True
    }:
        raise PreflightError("exact-zero key set does not match adapted reference records")
    fixture_ids = {key["fixture_id"] for key in expected_keys}
    if set(fixtures_by_id) != fixture_ids:
        raise PreflightError("fixture mapping has missing or unknown fixture IDs")
    for key in expected_keys:
        reference = references[key_tuple(key)]
        if reference.get("key") != key:
            raise PreflightError("reference mapping is not key-exact")
        fixture = fixtures_by_id[key["fixture_id"]]
        inputs = build_production_inputs(key, fixture)
        physical_zero = (
            parse_exact_fraction(fixture["density_kg_m3"]) == 0
            or parse_exact_fraction(
                fixture["displacement_psd_m2_hz"][inputs["frequency_index"]]
            ) == 0
        )
        if physical_zero != (key_tuple(key) in exact_zero_set):
            raise PreflightError("fixture zero semantics differ from frozen exact-zero key set")

    output = validate_output_parent(output_parent, repository_roots)
    output.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_path(output)
    if output.exists():
        raise PreflightError("comparison output became non-fresh during preflight")
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    _fsync_directory(output.parent)

    manifest = {
        "schema_version": 1,
        "status": "partial",
        "decision": None,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_binding_sha256": _sha256_bytes(Path(authorization_path).read_bytes()),
        "authorization_review_sha256": expected_review_sha256,
        "pins": binding["pins"],
        "expected_key_count": len(expected_ids),
        "expected_keys": expected_keys,
        "completed_key_count": 0,
        "records_file": "records.jsonl",
    }
    manifest_path = output / "manifest.json"
    try:
        _write_new_fsynced(manifest_path, _canonical_json_bytes(manifest))
        _write_new_fsynced(output / "records.jsonl", b"")
        _fsync_directory(output)
        if evaluator_factory is not None:
            evaluator = evaluator_factory()

        all_accepted = True
        for key in expected_keys:
            key_id = key_tuple(key)
            fixture = fixtures_by_id[key["fixture_id"]]
            input_payload = build_production_inputs(key, fixture)
            matrix = evaluator(key, fixture, input_payload)
            expected_units = _budget_units(budget, key["representation"])
            result = assess_one(
                key=key,
                fixture=fixture,
                reference=references[key_id],
                production_matrix=matrix,
                expected_units=expected_units,
                source_connections_passed=source_connection_results[key_id],
                exact_coverage_passed=True,
                budget=budget,
                kernel=kernel,
            )
            artifact_record = {
                "key": result["key"],
                "pins": binding["pins"],
                "inputs": input_payload["evidence"],
                "production": result["production_quantity"],
                "u_up_exact": fraction_to_exact_text(references[key_id]["u_up_exact"]),
                "u_matrix_exact": fraction_to_exact_text(references[key_id]["u_matrix_exact"]),
                "kernel_uncertainty_components": references[key_id]["kernel_uncertainty"]["components"],
                "difference_upper": result["assessment"].get("difference_upper"),
                "reference_norm_upper": result["assessment"].get("reference_norm_upper"),
                "budget_lower": result["assessment"].get("budget_lower"),
                "budget_upper": result["assessment"].get("budget_upper"),
                "u_matrix_kernel": result["u_matrix_kernel"],
                "oracle_fraction_condition": result["assessment"].get("oracle_fraction_condition"),
                "structure_conditions": result["structure_conditions"],
                "contract_conditions": result["contract_conditions"],
                "exact_zero_pass": result["exact_zero_pass"],
                "assessment": result["assessment"],
                "accepted": result["accepted"],
            }
            _append_fsynced(output / "records.jsonl", artifact_record)
            manifest["completed_key_count"] += 1
            all_accepted &= result["accepted"]
            _replace_manifest(manifest_path, manifest)

        if manifest["completed_key_count"] != len(expected_ids):
            raise PreflightError("comparison did not produce complete exact science coverage")
        manifest["status"] = "complete"
        manifest["decision"] = "PASS" if all_accepted else "FAIL"
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        _replace_manifest(manifest_path, manifest)
        return output
    except BaseException as exc:
        try:
            _append_fsynced(
                output / "failures.jsonl",
                {
                    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                    "completed_key_count": manifest["completed_key_count"],
                    "failure": _format_exception(exc),
                },
            )
            if manifest.get("status") != "complete":
                manifest["status"] = "partial"
                manifest["last_failure"] = _format_exception(exc)
                _replace_manifest(manifest_path, manifest)
        except BaseException:
            # The already-fsynced partial manifest and records remain the
            # recovery source if the filesystem itself also fails.
            pass
        raise


def _budget_units(budget: dict[str, Any], representation: str) -> str:
    units = budget.get("units")
    if not isinstance(units, dict) or representation not in units:
        raise PreflightError("frozen unit mapping lacks requested representation")
    value = units[representation]
    if not isinstance(value, str) or not value:
        raise PreflightError("frozen unit label is malformed")
    return value


def _git_output(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PreflightError(f"git preflight failed in {repo}: {args[0]}") from exc


def verify_clean_git_pin(repo: str | Path, expected_commit: str, expected_tree: str) -> None:
    repository = Path(repo)
    if _git_output(repository, "rev-parse", "HEAD") != expected_commit:
        raise PreflightError("repository commit pin mismatch")
    if _git_output(repository, "rev-parse", "HEAD^{tree}") != expected_tree:
        raise PreflightError("repository tree pin mismatch")
    if _git_output(repository, "status", "--porcelain"):
        raise PreflightError("pinned repository worktree is dirty")


def verify_git_source_hashes(
    repo: str | Path,
    commit: str,
    expected: dict[str, dict[str, str]],
) -> None:
    repository = Path(repo)
    for relative_path, pins in expected.items():
        if set(pins) != {"git_blob", "sha256"}:
            raise PreflightError("source blob pin schema mismatch")
        actual_blob = _git_output(repository, "rev-parse", f"{commit}:{relative_path}")
        try:
            source = subprocess.run(
                ["git", "-C", str(repository), "show", f"{commit}:{relative_path}"],
                check=True,
                capture_output=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            raise PreflightError(f"pinned source blob is missing: {relative_path}") from exc
        if actual_blob != pins["git_blob"] or _sha256_bytes(source) != pins["sha256"]:
            raise PreflightError(f"pinned source blob mismatch: {relative_path}")


def verify_external_file_hashes(base_repo: str | Path, expected: dict[str, str]) -> None:
    root = Path(base_repo)
    for relative_path, digest in expected.items():
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise PreflightError("external input hash is malformed")
        if sha256_file(root / relative_path) != digest:
            raise PreflightError(f"frozen external input hash mismatch: {relative_path}")


def resolve_frozen_source_path(
    production_repo: str | Path,
    relative_path: str,
    expected_hashes: dict[str, str],
) -> Path:
    """Resolve one frozen input under P and recheck the bytes that will be read."""
    if (
        not isinstance(relative_path, str)
        or relative_path not in FROZEN_SOURCE_PATHS
        or relative_path not in expected_hashes
    ):
        raise PreflightError("frozen source path is not in the pinned input inventory")
    expected_digest = expected_hashes[relative_path]
    if not isinstance(expected_digest, str) or not _SHA256.fullmatch(expected_digest):
        raise PreflightError("frozen source digest is malformed")

    root_input = Path(production_repo).absolute()
    _reject_symlink_path(root_input)
    try:
        root = root_input.resolve(strict=True)
        candidate = root / relative_path
        _reject_symlink_path(candidate)
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise PreflightError("frozen source cannot be resolved inside pinned production checkout") from exc
    if not resolved.is_file() or sha256_file(resolved) != expected_digest:
        raise PreflightError(f"frozen source hash mismatch: {relative_path}")
    return resolved


def verify_comparator_pin(
    comparator_repo: str | Path,
    *,
    policy_path: str | Path,
    expected_pins: dict[str, Any],
) -> None:
    """Bind the executing driver, adapter, policy, and clean checkout to review."""
    repo = Path(comparator_repo)
    try:
        verify_clean_git_pin(
            repo,
            expected_pins["comparator_commit"],
            expected_pins["comparator_tree"],
        )
        comparator_sources = expected_pins["comparator_source_blobs"]
        comparator_file_hashes = expected_pins["comparator_file_sha256"]
        policy_digest = expected_pins["policy_sha256"]
    except (KeyError, TypeError) as exc:
        raise PreflightError("comparator pin contract is incomplete") from exc
    expected_paths = {
        "tests/oracles/gen1/compare_stochastic_production.py",
        "tests/oracles/gen1/b1_exact_record_adapter.py",
    }
    if set(comparator_sources) != expected_paths or set(comparator_file_hashes) != expected_paths:
        raise PreflightError("comparator source inventory mismatch")
    verify_git_source_hashes(repo, expected_pins["comparator_commit"], comparator_sources)
    verify_external_file_hashes(repo, comparator_file_hashes)
    policy_path = Path(policy_path).absolute()
    try:
        policy_relative = policy_path.relative_to(repo.absolute()).as_posix()
    except ValueError as exc:
        raise PreflightError("comparator policy must be inside the pinned checkout") from exc
    if policy_relative != "tests/oracles/gen1/b1-production-comparator-policy-v1.json":
        raise PreflightError("unexpected comparator policy path")
    if not isinstance(policy_digest, str) or not _SHA256.fullmatch(policy_digest):
        raise PreflightError("policy digest pin is malformed")
    if sha256_file(policy_path) != policy_digest:
        raise PreflightError("comparator policy hash mismatch")


def validate_policy_schema(policy: dict[str, Any]) -> None:
    if set(policy) != _POLICY_FIELDS:
        missing = sorted(_POLICY_FIELDS - set(policy))
        unknown = sorted(set(policy) - _POLICY_FIELDS)
        raise PreflightError(f"comparator policy schema mismatch; missing={missing}, unknown={unknown}")
    if type(policy["schema_version"]) is not int or policy["schema_version"] != 1:
        raise PreflightError("comparator policy version mismatch")
    if policy["production_commit"] != PRODUCTION_COMMIT or policy["production_tree"] != PRODUCTION_TREE:
        raise PreflightError("production revision differs from fixed B1 candidate")
    if not isinstance(policy["production_repo"], str) or not policy["production_repo"]:
        raise PreflightError("production checkout path is missing")
    path_maps = (
        (policy["reference_paths"], {"review", "records", "seal"}, "reference path"),
        (policy["reference_sha256"], {"review", "records", "seal"}, "reference hash"),
    )
    for mapping, fields, label in path_maps:
        if not isinstance(mapping, dict) or set(mapping) != fields:
            raise PreflightError(f"{label} inventory mismatch")
    if policy["reference_sha256"] != REFERENCE_SHA256:
        raise PreflightError("accepted R_B1 artifact pin mismatch")
    if set(policy["production_git_blobs"]) != {
        "src/gragra/_arrays.py",
        "src/gragra/analytic/stochastic_bodywave.py",
        "src/gragra/constants.py",
        "src/gragra/observables/stochastic_bodywave.py",
        "src/gragra/waves/stochastic.py",
    }:
        raise PreflightError("production source blob inventory is missing")
    for pin in policy["production_git_blobs"].values():
        if (
            not isinstance(pin, dict)
            or set(pin) != {"git_blob", "sha256"}
            or not isinstance(pin["git_blob"], str)
            or not re.fullmatch(r"[0-9a-f]{40}", pin["git_blob"])
            or not isinstance(pin["sha256"], str)
            or not _SHA256.fullmatch(pin["sha256"])
        ):
            raise PreflightError("production source blob pin schema is malformed")
    if not isinstance(policy["frozen_source_sha256"], dict) or set(policy["frozen_source_sha256"]) != FROZEN_SOURCE_PATHS:
        raise PreflightError("the exact 14 frozen source digests are required")
    if any(not isinstance(value, str) or not _SHA256.fullmatch(value) for value in policy["frozen_source_sha256"].values()):
        raise PreflightError("frozen source digest inventory is malformed")
    for name in ("fixture_path", "budget_path", "procedure_path"):
        if not isinstance(policy[name], str) or not policy[name]:
            raise PreflightError(f"frozen input path is missing: {name}")
    if policy["fixture_path"] != "docs/design/gen1/stochastic/B1-fixtures.json" or policy["budget_path"] != "docs/design/gen1/stochastic/B1-budget.json" or policy["procedure_path"] != "docs/design/gen1/stochastic/B1-procedure-v2.json":
        raise PreflightError("unexpected B1 fixture, budget, or procedure path")
    if policy["g_si"] != "6.6743e-11":
        raise PreflightError("frozen G value is malformed")
    environment = policy["environment"]
    if not isinstance(environment, dict) or set(environment) != {
        "distributions",
        "binary_files",
        "site_packages_directory",
        "runtime",
        "binary_module_origins",
        "inventory_path",
        "inventory_sha256",
        "runtime_record_path",
        "runtime_record_sha256",
    }:
        raise PreflightError("environment policy schema mismatch")
    if not isinstance(environment["runtime"], dict) or not environment["runtime"]:
        raise PreflightError("runtime pin inventory is missing")
    if set(environment["runtime"]) != {
        "python_executable",
        "python_realpath",
        "python_version",
        "platform",
        "sys_path_tail",
        "distributions",
        "binary_sha256",
        "pth_sha256",
        "nnsus_imported",
        "gragra_imported_before_evaluator",
    }:
        raise PreflightError("runtime pin schema mismatch")
    if set(policy["production_modules"]) != {"gragra", "analytic", "observable", "constants"}:
        raise PreflightError("production module origin inventory mismatch")
    if set(environment["binary_files"]) != {"numpy_core_multiarray_binary", "scipy_special_ufuncs_binary"}:
        raise PreflightError("binary environment file inventory mismatch")
    if set(environment["binary_module_origins"]) != set(environment["binary_files"]):
        raise PreflightError("binary import origin inventory mismatch")
    if set(environment["distributions"]) != {"gragra", "mpmath", "nnsus", "numpy", "scipy", "sympy"}:
        raise PreflightError("runtime distribution inventory mismatch")
    if policy["production_api_parameters"] != _EXPECTED_API_PARAMETERS:
        raise PreflightError("production API inventory mismatch")
    if policy["exact_zero_fixture_ids"] != ["zero_density", "zero_psd"]:
        raise PreflightError("exact-zero fixture inventory mismatch")
    receipt_path = policy["source_connection_receipt_path"]
    receipt_digest = policy["source_connection_receipt_sha256"]
    if (receipt_path is None) != (receipt_digest is None):
        raise PreflightError("source-connection receipt path/hash must be pinned together")
    if receipt_digest is not None and (not isinstance(receipt_digest, str) or not _SHA256.fullmatch(receipt_digest)):
        raise PreflightError("source-connection receipt hash is malformed")
    pins = policy["authorization_pin_contract"]
    expected_pin_fields = {
        "comparator_source_blobs",
        "comparator_file_sha256",
        "production_commit",
        "production_tree",
        "production_git_blobs",
        "frozen_source_sha256",
        "reference_sha256",
        "kernel_git_blob",
        "kernel_sha256",
        "environment_inventory_sha256",
        "environment_runtime_sha256",
        "source_connection_receipt_sha256",
    }
    if not isinstance(pins, dict) or set(pins) != expected_pin_fields:
        raise PreflightError("authorization pin contract schema mismatch")
    static_expected = {
        "production_commit": policy["production_commit"],
        "production_tree": policy["production_tree"],
        "production_git_blobs": policy["production_git_blobs"],
        "frozen_source_sha256": policy["frozen_source_sha256"],
        "reference_sha256": policy["reference_sha256"],
        "kernel_git_blob": KERNEL_GIT_BLOB,
        "kernel_sha256": KERNEL_SHA256,
        "environment_inventory_sha256": environment["inventory_sha256"],
        "environment_runtime_sha256": environment["runtime_record_sha256"],
        "source_connection_receipt_sha256": receipt_digest,
    }
    for name, expected in static_expected.items():
        if pins[name] != expected:
            raise PreflightError(f"authorization pin contract differs from policy: {name}")
    for name in ("comparator_source_blobs", "comparator_file_sha256"):
        if not isinstance(pins[name], dict) or not pins[name]:
            raise PreflightError(f"authorization comparator file pins are missing: {name}")
    comparator_paths = {
        "tests/oracles/gen1/compare_stochastic_production.py",
        "tests/oracles/gen1/b1_exact_record_adapter.py",
    }
    if set(pins["comparator_source_blobs"]) != comparator_paths or set(pins["comparator_file_sha256"]) != comparator_paths:
        raise PreflightError("authorization comparator source set mismatch")
    for relative, source_pin in pins["comparator_source_blobs"].items():
        if (
            not isinstance(source_pin, dict)
            or set(source_pin) != {"git_blob", "sha256"}
            or not isinstance(source_pin["git_blob"], str)
            or not re.fullmatch(r"[0-9a-f]{40}", source_pin["git_blob"])
            or not isinstance(source_pin["sha256"], str)
            or not _SHA256.fullmatch(source_pin["sha256"])
            or pins["comparator_file_sha256"][relative] != source_pin["sha256"]
        ):
            raise PreflightError("authorization comparator source digest mismatch")


def verify_environment_inventory(policy: dict[str, Any]) -> None:
    """Revalidate pinned environment metadata and the five indexed inventory files."""
    env = policy["environment"]
    if sha256_file(env["inventory_path"]) != env["inventory_sha256"]:
        raise PreflightError("environment index hash mismatch")
    if sha256_file(env["runtime_record_path"]) != env["runtime_record_sha256"]:
        raise PreflightError("environment runtime record hash mismatch")
    index = _strict_cli_json(env["inventory_path"])
    runtime = _strict_cli_json(env["runtime_record_path"])
    if (
        index.get("environment") != "gragra-stoch-integration-20261002"
        or index.get("environment_mutation_performed") is not False
        or index.get("production_computation_performed") is not False
        or index.get("pinned_production_commit") != policy["production_commit"]
        or index.get("pinned_production_tree") != policy["production_tree"]
        or runtime.get("production_calls_performed") is not False
        or runtime.get("nnsus_module_imported") is not False
    ):
        raise PreflightError("environment inventory status or production pin mismatch")
    records = index.get("records")
    if not isinstance(records, dict) or set(records) != {
        "conda-explicit.txt",
        "conda-packages.json",
        "pip-freeze.txt",
        "pip-inspect.json",
        "runtime.json",
    }:
        raise PreflightError("environment inventory record set mismatch")
    for name, record in records.items():
        if not isinstance(record, dict) or not isinstance(record.get("sha256"), str):
            raise PreflightError("environment inventory record digest is malformed")
        if sha256_file(Path(env["inventory_path"]).parent / name) != record["sha256"]:
            raise PreflightError(f"environment inventory record hash mismatch: {name}")
    summary = index.get("runtime_summary")
    module_files = runtime.get("module_files")
    if not isinstance(summary, dict) or not isinstance(module_files, dict):
        raise PreflightError("environment runtime summary is malformed")
    if (
        env["runtime"].get("python_executable") != summary.get("python")
        or env["runtime"].get("python_realpath") != runtime.get("python_realpath")
        or env["runtime"].get("python_version") != summary.get("python_version")
        or env["runtime"].get("platform") != summary.get("platform")
        or env["runtime"].get("distributions") != summary.get("versions")
        or env["runtime"].get("pth_sha256") != summary.get("pth_sha256")
        or env["runtime"].get("sys_path_tail") != runtime.get("sys_path", [])[1:]
    ):
        raise PreflightError("policy runtime facts differ from the frozen inventory")
    expected_modules = {
        "gragra": "gragra",
        "analytic": "analytic",
        "observable": "observable",
        "constants": "constants",
    }
    for policy_name, inventory_name in expected_modules.items():
        if policy["production_modules"][policy_name] != module_files[inventory_name]:
            raise PreflightError(f"production module pin differs from environment record: {policy_name}")
    for binary_name, module_record_name in (
        ("numpy_core_multiarray_binary", "numpy_core_multiarray_binary"),
        ("scipy_special_ufuncs_binary", "scipy_special_ufuncs_binary"),
    ):
        if env["binary_module_origins"][binary_name] != module_files[module_record_name]:
            raise PreflightError(f"binary module origin differs from environment record: {binary_name}")


def collect_live_environment(policy: dict[str, Any]) -> dict[str, Any]:
    """Collect runtime facts without importing gragra or nnsus modules."""
    env = policy["environment"]
    versions = {}
    for distribution in env["distributions"]:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise PreflightError(f"pinned distribution is absent: {distribution}") from exc
    hashes = {}
    for name, path in env["binary_files"].items():
        hashes[name] = sha256_file(path)
    site_packages = Path(env["site_packages_directory"]).resolve(strict=True)
    if not site_packages.is_dir():
        raise PreflightError("pinned site-packages path is not a directory")
    pth_files: dict[str, str] = {}
    for path in sorted(site_packages.glob("*.pth")):
        if path.is_symlink() or not path.is_file():
            raise PreflightError("site-packages contains a non-regular .pth file")
        pth_files[path.name] = sha256_file(path)
    return {
        "python_executable": sys.executable,
        "python_realpath": os.path.realpath(sys.executable),
        "python_version": sys.version,
        "platform": platform.platform(),
        "sys_path_tail": sys.path[1:],
        "distributions": versions,
        "binary_sha256": hashes,
        "pth_sha256": pth_files,
        "nnsus_imported": "nnsus" in sys.modules,
        "gragra_imported_before_evaluator": any(
            name == "gragra" or name.startswith("gragra.") for name in sys.modules
        ),
    }


def verify_live_environment(policy: dict[str, Any]) -> dict[str, Any]:
    expected = policy["environment"]["runtime"]
    actual = collect_live_environment(policy)
    verify_pin_map(expected, actual)
    if actual["nnsus_imported"] or actual["gragra_imported_before_evaluator"]:
        raise PreflightError("unexpected ambient or production package was already imported")
    return actual


def _check_import_origin(module: ModuleType, expected_path: str, expected_sha256: str) -> None:
    path = Path(module.__file__ or "").resolve(strict=True)
    expected = Path(expected_path).resolve(strict=True)
    if path != expected or sha256_file(path) != expected_sha256:
        raise PreflightError(f"production module origin/hash mismatch: {module.__name__}")


def _production_evaluator_factory(policy: dict[str, Any]) -> Callable:
    """Import and pin public modules lazily, after durable partial setup."""
    if any(name == "gragra" or name.startswith("gragra.") for name in sys.modules):
        raise PreflightError("gragra was imported before the gated evaluator factory")
    analytic = importlib.import_module("gragra.analytic.stochastic_bodywave")
    observable = importlib.import_module("gragra.observables.stochastic_bodywave")
    constants = importlib.import_module("gragra.constants")
    gragra = importlib.import_module("gragra")
    for name, module in (
        ("gragra", gragra),
        ("analytic", analytic),
        ("observable", observable),
        ("constants", constants),
    ):
        pin = policy["production_modules"][name]
        _check_import_origin(module, pin["path"], pin["sha256"])
    import inspect

    for api_name, api in (
        ("gfree", analytic.fullspace_p_isotropic_acceleration_csd_g_free),
        ("si", observable.fullspace_p_isotropic_acceleration_csd),
    ):
        actual_parameters = {
            name: parameter.kind.name
            for name, parameter in inspect.signature(api).parameters.items()
        }
        if actual_parameters != policy["production_api_parameters"][api_name]:
            raise PreflightError(f"production API signature mismatch: {api_name}")

    for policy_name, module_name in (
        ("numpy_core_multiarray_binary", "numpy._core._multiarray_umath"),
        ("scipy_special_ufuncs_binary", "scipy.special._ufuncs"),
    ):
        module = importlib.import_module(module_name)
        pin = policy["environment"]["binary_module_origins"][policy_name]
        _check_import_origin(module, pin["path"], pin["sha256"])
    if any(name == "nnsus" or name.startswith("nnsus.") for name in sys.modules):
        raise PreflightError("nnsus was imported during production evaluator loading")
    expected_g = parse_exact_fraction(policy["g_si"])
    if float(expected_g).hex() != float(constants.G_SI).hex():
        raise PreflightError("runtime G_SI constant differs from frozen fixture pin")
    gfree = analytic.fullspace_p_isotropic_acceleration_csd_g_free
    si = observable.fullspace_p_isotropic_acceleration_csd

    def evaluate(key: dict[str, Any], fixture: dict[str, Any], inputs: dict[str, Any]) -> Any:
        api = si if key["representation"] == "SI" else gfree
        return api(**inputs["kwargs"])

    return evaluate


def _strict_cli_json(path: str | Path) -> dict[str, Any]:
    value = _read_json(path)
    if not isinstance(value, dict):
        raise PreflightError(f"expected JSON object: {path}")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--authorization-binding", required=True)
    parser.add_argument("--expected-authorization-sha256", required=True)
    parser.add_argument("--expected-review-sha256", required=True)
    parser.add_argument("--output-directory")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.execute == args.preflight_only:
        parser.error("select exactly one of --preflight-only or --execute")
    policy_path = Path(args.policy)
    policy = _strict_cli_json(policy_path)
    validate_policy_schema(policy)
    expected_pins = dict(policy["authorization_pin_contract"])
    expected_pins["policy_sha256"] = sha256_file(policy_path)
    binding = verify_authorization_bundle(
        args.authorization_binding,
        args.expected_authorization_sha256,
        args.expected_review_sha256,
        expected_pins,
    )

    comparator_repo = Path(__file__).resolve().parents[3]
    verify_comparator_pin(
        comparator_repo,
        policy_path=policy_path,
        expected_pins=binding["pins"],
    )
    production_repo = Path(policy["production_repo"])
    verify_clean_git_pin(production_repo, policy["production_commit"], policy["production_tree"])
    verify_git_source_hashes(
        production_repo,
        policy["production_commit"],
        policy["production_git_blobs"],
    )
    verify_external_file_hashes(production_repo, policy["frozen_source_sha256"])
    verify_environment_inventory(policy)
    verify_live_environment(policy)
    kernel = load_pinned_kernel(production_repo)

    reference_paths = policy["reference_paths"]
    if sha256_file(reference_paths["review"]) != policy["reference_sha256"]["review"]:
        raise PreflightError("accepted R_B1 review hash mismatch")
    if sha256_file(reference_paths["records"]) != policy["reference_sha256"]["records"]:
        raise PreflightError("sealed b1-002 records hash mismatch")
    if sha256_file(reference_paths["seal"]) != policy["reference_sha256"]["seal"]:
        raise PreflightError("b1-002 seal hash mismatch")

    review = _strict_cli_json(reference_paths["review"])
    if (
        review.get("R_B1") != "ACCEPTED_FROZEN_REFERENCE_SET"
        or review.get("review_id") != "R_B1-b1-002"
        or review.get("attempt", {}).get("attempt_id") != "b1-002"
        or review.get("attempt", {}).get("records_sha256") != policy["reference_sha256"]["records"]
        or review.get("attempt", {}).get("seal_sha256") != policy["reference_sha256"]["seal"]
        or review.get("coverage")
        != {
            "duplicate": 0,
            "expected": 30,
            "generated": 30,
            "missing": 0,
            "resolved": 30,
            "unique": 30,
            "unknown": 0,
            "unresolved": 0,
        }
        or review.get("P_science", {}).get("production_comparison_run") is not False
    ):
        raise PreflightError("accepted R_B1 review status, coverage, or no-comparison state mismatch")
    fixtures = _strict_cli_json(
        resolve_frozen_source_path(
            production_repo,
            policy["fixture_path"],
            policy["frozen_source_sha256"],
        )
    )
    budget = _strict_cli_json(
        resolve_frozen_source_path(
            production_repo,
            policy["budget_path"],
            policy["frozen_source_sha256"],
        )
    )
    procedure = _strict_cli_json(
        resolve_frozen_source_path(
            production_repo,
            policy["procedure_path"],
            policy["frozen_source_sha256"],
        )
    )
    gate_fields = procedure.get("connection_evaluator_contract", {}).get("gate_fields")
    expected_keys = fixtures["expected_keys"]
    exact_zero_keys = [
        key for key in expected_keys if key["fixture_id"] in set(policy["exact_zero_fixture_ids"])
    ]
    if len(expected_keys) != 30 or len(exact_zero_keys) != 12:
        raise PreflightError("frozen B1 key count or exact-zero partition mismatch")
    if len(review["uncertainty_and_oracle_fraction"]["per_record"]) != 30:
        raise PreflightError("accepted R_B1 sidecar key count mismatch")
    references = adapt_b1_records(
        Path(reference_paths["records"]).read_bytes(),
        review["uncertainty_and_oracle_fraction"]["per_record"],
        expected_keys,
        exact_zero_keys,
    )
    fixtures_by_id = {row["fixture_id"]: row for row in fixtures["science"]}
    if len(fixtures_by_id) != len(fixtures["science"]):
        raise PreflightError("duplicate science fixture ID")

    receipt_path = policy["source_connection_receipt_path"]
    receipt_sha = policy["source_connection_receipt_sha256"]
    if receipt_path is None or receipt_sha is None:
        raise PreflightError("source-connection receipt is absent; 70-key gate remains unresolved")
    if sha256_file(receipt_path) != receipt_sha:
        raise PreflightError("source-connection receipt hash mismatch")
    receipt = _strict_cli_json(receipt_path)
    keyed_connections = validate_source_connection_receipt(
        receipt,
        fixtures,
        production_commit=policy["production_commit"],
        production_tree=policy["production_tree"],
        fixture_sha256=policy["frozen_source_sha256"]["docs/design/gen1/stochastic/B1-fixtures.json"],
        gate_fields=gate_fields,
        expected_science_keys=expected_keys,
    )

    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "PREFLIGHT_PASS",
                    "authorization": binding["authorization_status"],
                    "production_comparison_run": False,
                    "science_key_count": len(expected_keys),
                    "exact_zero_key_count": len(exact_zero_keys),
                    "kernel_sha256": KERNEL_SHA256,
                },
                sort_keys=True,
            )
        )
        return 0

    if not args.output_directory:
        parser.error("--output-directory is required with --execute")
    return_path = execute_authorized_comparison(
        authorization_path=args.authorization_binding,
        expected_authorization_sha256=args.expected_authorization_sha256,
        expected_review_sha256=args.expected_review_sha256,
        expected_pins=expected_pins,
        expected_keys=expected_keys,
        exact_zero_keys=exact_zero_keys,
        fixtures_by_id=fixtures_by_id,
        references=references,
        evaluator_factory=lambda: _production_evaluator_factory(policy),
        kernel=kernel,
        budget={**budget, "units": fixtures["units"]},
        source_connection_results=keyed_connections,
        output_parent=args.output_directory,
        repository_roots=[production_repo, comparator_repo],
    )
    print(return_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
