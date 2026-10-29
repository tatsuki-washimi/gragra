"""Build and verify the frozen A2 two-axis diagnostic coverage map."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from decimal import Decimal, InvalidOperation
from typing import Any

FIXTURE_SHA256 = "8e5c5ba6f739bd978310b3087a12f966447d4e8e94427f14c776e6f4b1f020b7"
PILOT_MAP_SHA256 = "abc1c5ffef4140fa466db888dff4e1950de2377f376e4d626fed5957e9144bac"
NATIVE_KEY_FIELDS = (
    "family",
    "fixture_id",
    "frequency_hz",
    "target_id",
    "observable",
    "representation",
    "set",
)
PILOT_SCALAR_IDS = (
    "a2_depth_01",
    "a2_depth_09",
    "a2_depth_10",
    "a2_depth_11",
    "a2_depth_21",
    "a2_depth_27",
)
PILOT_PHYSICAL_IDS = ("a2_depth_10", "a2_depth_21", "a2_depth_27")
ITERATIONS = (0, 1, 2, 3)
FROZEN_PILOT_SCHEMA_FIELDS = {
    "classification",
    "coverage_map_sha256",
    "entries",
    "iterations",
    "physical_rule",
    "same_process",
    "scope",
}
FROZEN_PILOT_CLASSIFICATION = "Observed"
FROZEN_PILOT_SCOPE = "frozen diagnostic input selection, no runtime or science credit"
FROZEN_PILOT_PHYSICAL_RULE = "pass frozen physical tuple; never reconstruct from nominal x"


class CoverageError(ValueError):
    """A frozen key, classification, map pin or record set is inconsistent."""


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CoverageError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise CoverageError(f"nonfinite JSON number: {value}")


def strict_json(raw: bytes) -> Any:
    try:
        return json.loads(
            raw,
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CoverageError("malformed JSON bytes") from exc


def canonical_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError) as exc:
        raise CoverageError("value is not canonical JSON") from exc


def canonical_native_key(key: dict[str, Any]) -> str:
    if not isinstance(key, dict) or set(key) != set(NATIVE_KEY_FIELDS):
        raise CoverageError("native key fields differ from the frozen seven-field key")
    if any(not isinstance(key[name], str) or not key[name] for name in NATIVE_KEY_FIELDS):
        raise CoverageError("native key values must be nonempty strings")
    return json.dumps(
        key,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _finite_decimal(value: Any, label: str) -> None:
    if not isinstance(value, str):
        raise CoverageError(f"{label} must retain its frozen decimal spelling")
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise CoverageError(f"{label} is not a decimal string") from exc
    if not number.is_finite():
        raise CoverageError(f"{label} is nonfinite")


def validate_finite_decimal(value: Any, label: str = "value") -> str:
    """Public diagnostic guard for the fixture's exact finite decimal spelling."""
    _finite_decimal(value, label)
    return value


def _classify(key: dict[str, Any]) -> tuple[str, str]:
    if key["family"] != "A2" or key["set"] not in {"science", "auxiliary"}:
        raise CoverageError("unexpected A2 family or frozen set value")
    if key["observable"] == "acceleration_csd":
        if key["representation"] not in {"Gfree", "SI"}:
            raise CoverageError("physical CSD representation is not Gfree/SI")
        if key["target_id"] != "underground_origin":
            raise CoverageError("physical CSD target differs from frozen target")
        return "physical_csd", key["set"]
    if key["observable"] == "isotropic_depth_factor":
        if key["representation"] != "Gfree":
            raise CoverageError("scalar AUX representation must remain Gfree")
        if key["set"] != "auxiliary" or key["target_id"] != "nominal_x":
            raise CoverageError("scalar AUX key was promoted or retargeted")
        return "nominal_scalar", "auxiliary"
    raise CoverageError("unknown frozen A2 observable")


def build_map(fixture_bytes: bytes) -> dict[str, Any]:
    """Return a deterministic native-key map without importing A2 or Git code."""
    fixture_digest = hashlib.sha256(fixture_bytes).hexdigest()
    if fixture_digest != FIXTURE_SHA256:
        raise CoverageError("frozen A2 fixture hash differs")
    fixture = strict_json(fixture_bytes)
    if not isinstance(fixture, dict) or fixture.get("family") != "A2":
        raise CoverageError("frozen fixture is not A2")
    physical_keys = fixture.get("expected_keys")
    scalar_keys = fixture.get("expected_scalar_keys")
    rows = fixture.get("fixtures")
    if (
        not isinstance(physical_keys, list)
        or len(physical_keys) != 56
        or not isinstance(scalar_keys, list)
        or len(scalar_keys) != 28
        or not isinstance(rows, list)
        or len(rows) != 28
    ):
        raise CoverageError("frozen A2 key/row counts differ from the 84-key contract")

    all_keys = physical_keys + scalar_keys
    canonical_keys = [canonical_native_key(key) for key in all_keys]
    if len(set(canonical_keys)) != 84:
        raise CoverageError("frozen A2 native keys are duplicated")
    row_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("key_base"), dict):
            raise CoverageError("malformed frozen fixture row")
        fixture_id = row["key_base"].get("fixture_id")
        if not isinstance(fixture_id, str) or fixture_id in row_by_id:
            raise CoverageError("frozen fixture row id is absent or duplicated")
        if row.get("purpose") != "formal-science":
            raise CoverageError("frozen fixture purpose changed")
        if row.get("pressure_convention") != "surface_total":
            raise CoverageError("frozen surface-total pressure convention changed")
        if row.get("psd_convention") != "surface_total_one_sided_per_hz":
            raise CoverageError("frozen one-sided per-Hz convention changed")
        inputs = row.get("inputs")
        if not isinstance(inputs, dict) or set(inputs) != {
            "depth_m",
            "frequency_hz",
            "g_si",
            "pressure_psd_pa2_hz",
            "sound_speed_m_s",
        }:
            raise CoverageError("frozen physical input tuple is incomplete")
        for name, value in inputs.items():
            _finite_decimal(value, f"{fixture_id}.{name}")
        for name in ("actual_physical_x", "nominal_exact_x"):
            _finite_decimal(row.get(name), f"{fixture_id}.{name}")
        row_by_id[fixture_id] = row

    if set(row_by_id) != {key["fixture_id"] for key in all_keys}:
        raise CoverageError("fixture rows and frozen expected keys are not bijective")

    entries: list[dict[str, Any]] = []
    used: set[str] = set()
    for row in rows:
        fixture_id = row["key_base"]["fixture_id"]
        row_keys = [key for key in physical_keys if key["fixture_id"] == fixture_id]
        row_keys += [key for key in scalar_keys if key["fixture_id"] == fixture_id]
        if len(row_keys) != 3:
            raise CoverageError(f"{fixture_id} must map to two physical keys and one scalar key")
        for key in row_keys:
            encoded_key = canonical_native_key(key)
            if encoded_key in used:
                raise CoverageError("duplicate native key in frozen row map")
            used.add(encoded_key)
            kind, role = _classify(key)
            if key["frequency_hz"] != row["inputs"]["frequency_hz"]:
                raise CoverageError("key frequency and exact fixture input differ")
            if kind == "physical_csd":
                base = row["key_base"]
                for name in ("family", "fixture_id", "frequency_hz", "observable", "set", "target_id"):
                    if key[name] != base[name]:
                        raise CoverageError("physical key differs from frozen row key base")
            else:
                if key["fixture_id"] != fixture_id:
                    raise CoverageError("scalar key and fixture row differ")
            entries.append(
                {
                    "acceptance_role": role,
                    "actual_physical_x": row["actual_physical_x"],
                    "canonical_key_json": encoded_key,
                    "fixture_row_sha256": hashlib.sha256(canonical_bytes(row)).hexdigest(),
                    "inputs": dict(row["inputs"]),
                    "native_key": dict(key),
                    "nominal_exact_x": row["nominal_exact_x"],
                    "record_kind": kind,
                }
            )
    if len(entries) != 84 or used != set(canonical_keys):
        raise CoverageError("fixture/map/native key bijection failed")
    count = Counter((entry["acceptance_role"], entry["record_kind"]) for entry in entries)
    expected_counts = {
        ("science", "physical_csd"): 54,
        ("auxiliary", "physical_csd"): 2,
        ("auxiliary", "nominal_scalar"): 28,
    }
    if count != expected_counts:
        raise CoverageError("A2 two-axis key counts differ from the frozen contract")
    return {
        "schema": "a2-v2-two-axis-map-v1",
        "fixture_sha256": fixture_digest,
        "key_fields": list(NATIVE_KEY_FIELDS),
        "counts": [
            {"acceptance_role": role, "count": number, "record_kind": kind}
            for (role, kind), number in sorted(count.items())
        ],
        "entries": entries,
        "scope": "diagnostic_key_classification_only",
    }


def verify_map_artifact(
    fixture_map: dict[str, Any], map_bytes: bytes, expected_sha256: str = PILOT_MAP_SHA256
) -> dict[str, Any]:
    """Bind the reviewed observed map bytes to a fresh deterministic fixture map."""
    actual_sha256 = hashlib.sha256(map_bytes).hexdigest()
    if actual_sha256 != expected_sha256 or expected_sha256 != PILOT_MAP_SHA256:
        raise CoverageError("coverage-map artifact pin differs")
    observed = strict_json(map_bytes)
    if (
        not isinstance(observed, dict)
        or observed.get("classification") != "Observed"
        or observed.get("fixture_sha256") != FIXTURE_SHA256
        or observed.get("scope") != "static_fixture_classification_only"
        or observed.get("entries") != fixture_map.get("entries")
        or observed.get("row_encoding")
        != "compact sorted-key JSON UTF-8 terminal LF"
        or observed.get("key_encoding")
        != "compact sorted-key native JSON UTF-8, terminal LF for hash; original native key strings preserved"
    ):
        raise CoverageError("observed 84-key map differs from the rebuilt frozen map")
    if observed.get("counts") != fixture_map.get("counts"):
        raise CoverageError("observed two-axis counts differ from rebuilt frozen map")
    for entry in observed["entries"]:
        canonical_native_key(entry.get("native_key"))
        if entry.get("canonical_key_json") != canonical_native_key(entry["native_key"]):
            raise CoverageError("observed native-key encoding differs")
    return {
        "sha256": actual_sha256,
        "record_count": len(observed["entries"]),
        "counts": observed["counts"],
        "entries": observed["entries"],
    }


def select_pilot(
    fixture_map: dict[str, Any], *, coverage_map_sha256: str
) -> list[dict[str, Any]]:
    """Select the frozen 12 keys and preserve physical tuples independently of x."""
    if coverage_map_sha256 != PILOT_MAP_SHA256:
        raise CoverageError("pilot must consume the exact pinned 84-key map")
    entries = fixture_map.get("entries")
    if not isinstance(entries, list) or len(entries) != 84:
        raise CoverageError("pilot input map does not contain all 84 native keys")
    selected: list[dict[str, Any]] = []

    for fixture_id in PILOT_SCALAR_IDS:
        match = [
            entry
            for entry in entries
            if entry["native_key"]["fixture_id"] == fixture_id
            and entry["record_kind"] == "nominal_scalar"
        ]
        if len(match) != 1:
            raise CoverageError("fixed scalar pilot key is missing or duplicated")
        entry = match[0]
        selected.append(
            {
                **entry,
                "input_mode": "nominal_x",
                "iterations": list(ITERATIONS),
                "same_process": True,
            }
        )

    for fixture_id in PILOT_PHYSICAL_IDS:
        for representation in ("Gfree", "SI"):
            match = [
                entry
                for entry in entries
                if entry["native_key"]["fixture_id"] == fixture_id
                and entry["record_kind"] == "physical_csd"
                and entry["native_key"]["representation"] == representation
                and entry["acceptance_role"] == "science"
            ]
            if len(match) != 1:
                raise CoverageError("fixed physical pilot key is missing or duplicated")
            entry = match[0]
            copied = dict(entry)
            copied.pop("nominal_exact_x", None)
            selected.append(
                {
                    **copied,
                    "input_mode": "physical_tuple",
                    "iterations": list(ITERATIONS),
                    "same_process": True,
                }
            )
    if len(selected) != 12 or len({entry["canonical_key_json"] for entry in selected}) != 12:
        raise CoverageError("fixed pilot is not a 12-key native-key set")
    return selected


def adapt_frozen_pilot_selection(
    raw: bytes,
    fixture_map: dict[str, Any],
    *,
    coverage_map_sha256: str,
    expected_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Adapt the pinned observed artifact to execution entries without rewriting it.

    The observed file stores iterations and same-process policy once at the top
    level and retains nominal_exact_x metadata even for physical rows. The
    execution form expands those top-level fields per entry and removes that
    scalar metadata from physical call inputs.
    """
    if not isinstance(raw, bytes) or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise CoverageError("frozen pilot bytes differ from the exact external pin")
    observed = strict_json(raw)
    if (
        not isinstance(observed, dict)
        or set(observed) != FROZEN_PILOT_SCHEMA_FIELDS
        or observed.get("classification") != FROZEN_PILOT_CLASSIFICATION
        or observed.get("scope") != FROZEN_PILOT_SCOPE
        or observed.get("coverage_map_sha256") != coverage_map_sha256
        or observed.get("iterations") != list(ITERATIONS)
        or observed.get("same_process") is not True
        or observed.get("physical_rule") != FROZEN_PILOT_PHYSICAL_RULE
        or not isinstance(observed.get("entries"), list)
        or len(observed["entries"]) != 12
    ):
        raise CoverageError("frozen pilot artifact differs from its exact observed schema")

    execution_entries = select_pilot(fixture_map, coverage_map_sha256=coverage_map_sha256)
    map_entries = fixture_map.get("entries")
    if not isinstance(map_entries, list) or len(map_entries) != 84:
        raise CoverageError("frozen pilot adaptation requires the complete 84-key map")
    by_key = {entry.get("canonical_key_json"): entry for entry in map_entries if isinstance(entry, dict)}
    if len(by_key) != 84:
        raise CoverageError("coverage map has duplicate or malformed keys during pilot adaptation")
    if len(execution_entries) != len(observed["entries"]):
        raise CoverageError("frozen and execution pilot entry counts differ")

    kind_counts = {"nominal_scalar": 0, "physical_csd": 0}
    for frozen_entry, execution_entry in zip(observed["entries"], execution_entries):
        if not isinstance(frozen_entry, dict):
            raise CoverageError("frozen pilot entry is not an object")
        source_entry = by_key.get(execution_entry.get("canonical_key_json"))
        if source_entry is None or frozen_entry != source_entry:
            raise CoverageError("frozen pilot row differs from its exact 84-key source-map row")
        kind = execution_entry.get("record_kind")
        expected_mode = "nominal_x" if kind == "nominal_scalar" else "physical_tuple"
        if (
            kind not in kind_counts
            or execution_entry.get("input_mode") != expected_mode
            or execution_entry.get("iterations") != list(ITERATIONS)
            or execution_entry.get("same_process") is not True
        ):
            raise CoverageError("pilot adaptation produced an invalid execution convention")
        if kind == "physical_csd":
            if "nominal_exact_x" in execution_entry or "nominal_exact_x" not in frozen_entry:
                raise CoverageError("physical pilot metadata was confused with its physical input tuple")
        elif execution_entry.get("nominal_exact_x") != frozen_entry.get("nominal_exact_x"):
            raise CoverageError("scalar pilot adaptation changed its native nominal x")
        kind_counts[kind] += 1
    if kind_counts != {"nominal_scalar": 6, "physical_csd": 6}:
        raise CoverageError("frozen pilot adaptation changed its fixed six/six selection")
    return execution_entries, {
        "adapter_schema": "gragra.a2.v2.frozen-pilot-adapter.v1",
        "source_artifact_sha256": expected_sha256,
        "coverage_map_sha256": coverage_map_sha256,
        "input_row_count": len(observed["entries"]),
        "execution_entry_count": len(execution_entries),
        "iterations": list(ITERATIONS),
        "same_process": True,
        "physical_nominal_exact_x": "preserved_as_source_metadata_not_execution_input",
    }


def verify_records(fixture_map: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    """Require a complete exact native-key and frozen-input bijection."""
    if not isinstance(records, list) or not isinstance(fixture_map.get("entries"), list):
        raise CoverageError("record and expected map collections must be lists")
    expected = {entry["canonical_key_json"]: entry for entry in fixture_map["entries"]}
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise CoverageError("record must be an object")
        key = record.get("key", record.get("native_key"))
        encoded_key = canonical_native_key(key)
        if encoded_key not in expected:
            raise CoverageError("record contains an unknown native key")
        if encoded_key in seen:
            raise CoverageError("record contains a duplicate native key")
        seen.add(encoded_key)
        entry = expected[encoded_key]
        if entry["record_kind"] == "physical_csd":
            expected_inputs = entry["inputs"]
        elif entry["record_kind"] == "nominal_scalar":
            expected_inputs = {
                "frequency_hz": entry["native_key"]["frequency_hz"],
                "nominal_exact_x": entry["nominal_exact_x"],
            }
        else:
            raise CoverageError("frozen map contains an unknown record kind")
        if record.get("inputs") != expected_inputs:
            raise CoverageError("record kind-specific exact decimal inputs differ from frozen fixture")
        if "record_kind" in record and record["record_kind"] != entry["record_kind"]:
            raise CoverageError("record kind differs from the independent map axis")
        if "acceptance_role" in record and record["acceptance_role"] != entry["acceptance_role"]:
            raise CoverageError("acceptance role differs from the frozen key set")
    missing = set(expected) - seen
    if missing:
        raise CoverageError(f"record set is missing {len(missing)} native keys")
    return {
        "record_count": len(records),
        "unique_native_keys": len(seen),
        "coverage_status": "complete_diagnostic_inventory",
        "science_credit": 0,
    }
