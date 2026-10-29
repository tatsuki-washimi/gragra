"""Validate the precision-preserving 48^3 cosine-bell OpenFOAM export.

The native OpenFOAM parser in this module is an export audit only. All mass,
gravity, oracle, translation, successor, and sign calculations use the
unchanged ``load_openfoam_density`` snapshots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from gragra.adapters.openfoam_density import load_openfoam_density
from gragra.observables import (
    point_acceleration,
    point_gravity_gradient,
    point_potential,
)
from scripts.research_openfoam_cosine_bell import (
    ANALYSIS_TIMEOUT_S,
    MAX_RETAINED_BYTES,
    MIN_FREE_BYTES,
    PROVENANCE_TEMPLATE,
    TARGETS_M,
    TIMES_S,
    _all_retained_run_bytes,
    _discrete_scales,
    _norm_per_target,
    _sha256,
    _time_limit,
    analytic_total_mass_kg,
    centre_m,
    literal_discrete_observables,
    point_anomaly_observables,
)

LEVEL = 48
CELL_COUNT = LEVEL**3
POINT_COUNT = (LEVEL + 1) ** 3
TIMES = tuple(format(value, "g") for value in TIMES_S)
PRIOR_LEVEL_HASHES = {
    "16": "9675b26abbae0e154f2f1061ab025482fd0854ec41b443f65bee1936c4280292",
    "32": "523e93cd9ed5d8cfc305dcd8befda99709d314eb72ffa240fe9749e508ed8432",
}
PRIOR_PRIMARY_EVIDENCE_HASH = (
    "da6aaa25ec5f8a8d44394d44c8c8dbeb54085e78c894ac12b6f2f05ee946635f"
)
BASE_SHA = "c9ceda3e93e9cea0bb34ec5dc4894f248b08c317"
_NUMBER = re.compile(
    r"[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|inf(?:inity)?|nan)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class NativeArray:
    """Validated values from one native OpenFOAM ASCII list."""

    class_name: str
    object_name: str
    dimensions: tuple[int, ...] | None
    declared_count: int
    values: np.ndarray


def _read_ascii(path: Path) -> str:
    try:
        text = Path(path).read_text(encoding="ascii")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path}: native input must be ASCII") from error
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", text)


def _declaration(text: str, name: str, path: Path) -> str:
    match = re.search(rf"\b{re.escape(name)}\s+([^;]+);", text)
    if match is None:
        raise ValueError(f"{path}: missing {name} declaration")
    return match.group(1).strip()


def _parenthesized_body(text: str, opening: int, path: Path) -> tuple[str, int]:
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return text[opening + 1 : index], index
    raise ValueError(f"{path}: unterminated list payload")


def _numbers(text: str, path: Path) -> np.ndarray:
    tokens = _NUMBER.findall(text)
    remaining = _NUMBER.sub("", text)
    if re.sub(r"[\s()]+", "", remaining):
        raise ValueError(f"{path}: unexpected token in numeric payload")
    try:
        values = np.asarray([float(token) for token in tokens], dtype=np.float64)
    except ValueError as error:
        raise ValueError(f"{path}: invalid numeric payload") from error
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: payload must contain only finite values")
    return values


def _validate_header(
    text: str, path: Path, *, expected_class: str, expected_object: str
) -> tuple[str, str]:
    if _declaration(text, "format", path) != "ascii":
        raise ValueError(f"{path}: format must be ascii")
    class_name = _declaration(text, "class", path)
    object_name = _declaration(text, "object", path)
    if class_name != expected_class:
        raise ValueError(
            f"{path}: class must be {expected_class}, got {class_name}"
        )
    if object_name != expected_object:
        raise ValueError(
            f"{path}: object must be {expected_object}, got {object_name}"
        )
    return class_name, object_name


def _parse_native_points(path: Path, *, expected_count: int) -> NativeArray:
    """Parse the bounded native ``constant/polyMesh/points`` ASCII profile."""
    path = Path(path)
    text = _read_ascii(path)
    class_name, object_name = _validate_header(
        text, path, expected_class="vectorField", expected_object="points"
    )
    header_end = text.find("}")
    match = re.search(r"\b(\d+)\s*\(", text[header_end + 1 :])
    if match is None:
        raise ValueError(f"{path}: missing points count or payload")
    count = int(match.group(1))
    if count != expected_count:
        raise ValueError(
            f"{path}: declared count {count} does not match {expected_count}"
        )
    opening = header_end + 1 + match.end() - 1
    body, _closing = _parenthesized_body(text, opening, path)
    values = _numbers(body, path)
    if values.size != count * 3:
        raise ValueError(
            f"{path}: points component count {values.size} does not match {count * 3}"
        )
    return NativeArray(class_name, object_name, None, count, values.reshape(count, 3))


def _parse_native_field(
    path: Path,
    *,
    object_name: str,
    components: int,
    expected_count: int,
    expected_dimensions: tuple[int, ...],
) -> NativeArray:
    """Parse one bounded native volScalarField or volVectorField list."""
    if components not in {1, 3}:
        raise ValueError("components must be 1 or 3")
    path = Path(path)
    text = _read_ascii(path)
    expected_class = "volScalarField" if components == 1 else "volVectorField"
    class_name, declared_object = _validate_header(
        text, path, expected_class=expected_class, expected_object=object_name
    )
    dimensions_text = _declaration(text, "dimensions", path)
    dimensions_match = re.fullmatch(r"\[([^]]+)\]", dimensions_text)
    if dimensions_match is None:
        raise ValueError(f"{path}: invalid dimensions declaration")
    dimensions_values = _numbers(dimensions_match.group(1), path)
    if np.any(dimensions_values != np.floor(dimensions_values)):
        raise ValueError(f"{path}: dimensions must be integers")
    dimensions = tuple(int(value) for value in dimensions_values)
    if dimensions != expected_dimensions:
        raise ValueError(
            f"{path}: dimensions {dimensions} do not match {expected_dimensions}"
        )
    list_type = "scalar" if components == 1 else "vector"
    match = re.search(
        rf"\binternalField\s+nonuniform\s+List<{list_type}>\s+(\d+)\s*\(",
        text,
    )
    if match is None:
        raise ValueError(f"{path}: missing bounded internalField list")
    count = int(match.group(1))
    if count != expected_count:
        raise ValueError(
            f"{path}: declared count {count} does not match {expected_count}"
        )
    body, _closing = _parenthesized_body(text, match.end() - 1, path)
    values = _numbers(body, path)
    expected_values = count * components
    if values.size != expected_values:
        raise ValueError(
            f"{path}: payload component count {values.size} does not match "
            f"{expected_values}"
        )
    if components == 3:
        values = values.reshape(count, 3)
    return NativeArray(class_name, declared_object, dimensions, count, values)


def _assert_native_roundtrip(
    snapshot: Any,
    *,
    native_points: np.ndarray,
    native_centres: np.ndarray,
    native_volumes: np.ndarray,
    native_density: np.ndarray,
    expected_count: int,
) -> dict[str, object]:
    """Assert reader payloads equal native decimals rounded through float32."""
    expected = {
        "points": np.asarray(native_points, dtype=np.float32).astype(np.float64),
        "C": np.asarray(native_centres, dtype=np.float32).astype(np.float64),
        "Vc": np.asarray(native_volumes, dtype=np.float32).astype(np.float64),
        "delta_rho": np.asarray(native_density, dtype=np.float32).astype(np.float64),
    }
    observed = {
        "points": snapshot.mesh.vertices_m,
        "C": snapshot.positions_m,
        "Vc": snapshot.volumes_m3,
        "delta_rho": snapshot.density_kg_m3,
    }
    for name in expected:
        if not np.array_equal(observed[name], expected[name]):
            raise AssertionError(
                f"reader {name} differs from native ASCII rounded through float32"
            )
    ids = np.arange(expected_count, dtype=np.int64)
    if not np.array_equal(snapshot.cell_ids, ids):
        raise AssertionError(
            f"reader cellID must be the ordered range 0..{expected_count - 1}"
        )
    return {
        "passed": True,
        "comparison": "array_equal(reader, float32(native_ASCII).astype(float64))",
        "cell_ids": f"ordered 0..{expected_count - 1}",
        "points_compared_to": "snapshot.mesh.vertices_m",
    }


def summarize_continuum_errors(
    time_records: dict[str, dict[str, Any]], observable: str
) -> dict[str, object]:
    """Preserve every time/target error and identify the global maximum."""
    per_time = {
        time_name: list(time_records[time_name][observable]) for time_name in TIMES
    }
    matrix = np.asarray([per_time[time_name] for time_name in TIMES], dtype=np.float64)
    if matrix.shape != (3, len(TARGETS_M)) or not np.isfinite(matrix).all():
        raise ValueError("continuum errors must cover all three times and targets")
    flat_index = int(np.argmax(matrix))
    time_index, target_index = np.unravel_index(flat_index, matrix.shape)
    return {
        "per_time_target": per_time,
        "maximum": float(matrix[time_index, target_index]),
        "argmax": {
            "time_name": TIMES[time_index],
            "target_index": int(target_index),
        },
    }


def evaluate_gates(max_errors: dict[str, dict[str, float]]) -> dict[str, bool]:
    """Evaluate exactly the eight approved E48 threshold/E16 gates."""
    gates: dict[str, bool] = {}
    for observable, percent, threshold in (
        ("mass", 1, 0.01),
        ("potential", 2, 0.02),
        ("acceleration", 2, 0.02),
        ("gradient", 2, 0.02),
    ):
        values = max_errors[observable]
        e16 = float(values["16"])
        e48 = float(values["48"])
        if not np.isfinite([e16, e48]).all():
            raise ValueError(f"{observable} gate errors must be finite")
        gates[f"{observable}_E48_le_{percent}_percent"] = e48 <= threshold
        gates[f"{observable}_E48_lt_E16"] = e48 < e16
    return gates


def classify_scientific_status(
    gates: dict[str, bool], supporting_checks: dict[str, bool]
) -> str:
    """Return the O status after all measured gates and support checks."""
    return (
        "complete"
        if all(gates.values()) and all(supporting_checks.values())
        else "partial"
    )


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _retained_science_bytes(prior_run: Path, case: Path, science: Path) -> int:
    """Count every retained prior O run plus the new case and science outputs."""
    return (
        _all_retained_run_bytes(Path(prior_run).parent)
        + _tree_bytes(Path(case))
        + _tree_bytes(Path(science))
    )


def _manifest_case(manifest: dict[str, Any]) -> Path:
    case = manifest.get("case")
    if not isinstance(case, dict) or not isinstance(case.get("path"), str):
        raise ValueError("export manifest case.path must be a string")
    return Path(case["path"]).resolve()


def _verify_export_manifest(
    path: Path, case: Path
) -> tuple[dict[str, Any], dict[str, object], dict[str, str]]:
    manifest = _json(path)
    if manifest.get("status") != "complete":
        raise ValueError("export manifest status is not complete")
    if _manifest_case(manifest) != case.resolve():
        raise ValueError("--case differs from export manifest case")
    vtk_records = manifest.get("vtk_outputs")
    export_runs = manifest.get("export_runs")
    if not isinstance(vtk_records, dict) or set(vtk_records) != set(TIMES):
        raise ValueError("vtk_outputs must contain exactly times 0, 1, and 2")
    if not isinstance(export_runs, list):
        raise ValueError("export_runs must be a list")
    runs_by_time: dict[str, dict[str, Any]] = {}
    for run in export_runs:
        if not isinstance(run, dict) or run.get("time_name") not in TIMES:
            raise ValueError("each export run must declare time_name 0, 1, or 2")
        time_name = run["time_name"]
        if time_name in runs_by_time:
            raise ValueError(f"duplicate export run for time {time_name}")
        runs_by_time[time_name] = run
    if set(runs_by_time) != set(TIMES):
        raise ValueError("export_runs must bind exactly one run to each time")

    outputs: dict[str, object] = {}
    commands: dict[str, str] = {}
    for time_name in TIMES:
        vtk_path = case / "VTK" / f"{case.name}_{time_name}.vtk"
        if not vtk_path.is_file():
            raise FileNotFoundError(vtk_path)
        record = vtk_records[time_name]
        if not isinstance(record, dict):
            raise ValueError(f"vtk_outputs[{time_name}] must be an object")
        if Path(str(record.get("path"))).resolve() != vtk_path:
            raise ValueError(f"vtk output path mismatch for time {time_name}")
        if record.get("size_bytes") != vtk_path.stat().st_size:
            raise ValueError(f"vtk output size mismatch for time {time_name}")
        recorded_hash = record.get("sha256")
        observed_hash = _sha256(vtk_path)
        if recorded_hash != observed_hash:
            raise ValueError(f"export manifest hash mismatch for {vtk_path.name}")
        run = runs_by_time[time_name]
        utility_argv = run.get("utility_argv")
        if not (
            isinstance(utility_argv, list)
            and utility_argv
            and all(isinstance(item, str) for item in utility_argv)
            and Path(utility_argv[0]).name == "foamToVTK"
        ):
            raise ValueError(f"invalid utility_argv for time {time_name}")
        try:
            time_index = utility_argv.index("-time")
            case_index = utility_argv.index("-case")
        except ValueError as error:
            raise ValueError("utility_argv must include -case and -time") from error
        if utility_argv[time_index + 1] != time_name:
            raise ValueError(f"utility_argv time mismatch for {time_name}")
        if Path(utility_argv[case_index + 1]).resolve() != case:
            raise ValueError(f"utility_argv case mismatch for {time_name}")
        if Path(str(run.get("output_path"))).resolve() != vtk_path:
            raise ValueError(f"export run output mismatch for {time_name}")
        binding = record.get("export_run")
        if not isinstance(binding, dict):
            raise ValueError(f"vtk output lacks export_run binding for {time_name}")
        for key in ("name", "time_name", "utility_argv", "output_path"):
            if binding.get(key) != run.get(key):
                raise ValueError(f"vtk export_run {key} mismatch for {time_name}")
        if not isinstance(binding.get("argv_log"), dict):
            raise ValueError(f"vtk export_run lacks argv_log for {time_name}")
        commands[time_name] = shlex.join(utility_argv)
        outputs[time_name] = {
            "path": str(vtk_path),
            "bytes": vtk_path.stat().st_size,
            "sha256": observed_hash,
            "export_run": binding,
        }
    return manifest, outputs, commands


def _verify_prior(prior: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    levels: dict[str, Any] = {}
    for level, expected_hash in PRIOR_LEVEL_HASHES.items():
        path = prior / f"scientific-level-n{level}.json"
        if _sha256(path) != expected_hash:
            raise ValueError(f"prior scientific-level-n{level}.json hash mismatch")
        value = _json(path)
        if value.get("status") != "complete":
            raise ValueError(f"prior level {level} is not complete")
        levels[level] = value
    primary_path = prior / "scientific-manifest.json"
    if _sha256(primary_path) != PRIOR_PRIMARY_EVIDENCE_HASH:
        raise ValueError("prior primary scientific evidence hash mismatch")
    primary = _json(primary_path)
    derivation = primary.get("derivation", {})
    required = (
        "mass_exact_passed",
        "quadrature_passed",
        "advection_exact_passed",
        "boundary_continuity_passed",
    )
    if not all(derivation.get(key) is True for key in required):
        raise ValueError(
            "prior CAS/quadrature/advection/boundary evidence did not pass"
        )
    return levels, primary


def _prior_maxima(levels: dict[str, Any]) -> dict[str, dict[str, float]]:
    maxima = {name: {} for name in ("mass", "potential", "acceleration", "gradient")}
    for level, level_data in levels.items():
        times = level_data["times"]
        maxima["mass"][level] = max(
            float(times[time_name]["mass_relative_error"]) for time_name in TIMES
        )
        for observable in ("potential", "acceleration", "gradient"):
            maxima[observable][level] = max(
                max(times[time_name]["continuum_relative_errors"][observable])
                for time_name in TIMES
            )
    return maxima


def _provenance(export_command: str) -> dict[str, object]:
    provenance = {**PROVENANCE_TEMPLATE, "export_command": export_command}
    if len(provenance) != 9:
        raise AssertionError("loader producer profile must have exactly nine keys")
    return provenance


def _analyse_48(case: Path, export_commands: dict[str, str]) -> dict[str, Any]:
    process_rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    started = time.monotonic()
    points = _parse_native_points(
        case / "constant/polyMesh/points", expected_count=POINT_COUNT
    )
    snapshots = []
    observations: dict[str, Any] = {}
    with _time_limit(ANALYSIS_TIMEOUT_S):
        for time_s, time_name in zip(TIMES_S, TIMES, strict=True):
            centres = _parse_native_field(
                case / time_name / "C",
                object_name="C",
                components=3,
                expected_count=CELL_COUNT,
                expected_dimensions=(0, 1, 0, 0, 0, 0, 0),
            )
            volumes = _parse_native_field(
                case / time_name / "Vc",
                object_name="Vc",
                components=1,
                expected_count=CELL_COUNT,
                expected_dimensions=(0, 3, 0, 0, 0, 0, 0),
            )
            density = _parse_native_field(
                case / time_name / "delta_rho",
                object_name="delta_rho",
                components=1,
                expected_count=CELL_COUNT,
                expected_dimensions=(1, -3, 0, 0, 0, 0, 0),
            )
            vtk_path = case / "VTK" / f"{case.name}_{time_name}.vtk"
            snapshot = load_openfoam_density(
                vtk_path,
                field_name="delta_rho",
                time_s=time_s,
                time_name=time_name,
                coordinate_frame="openfoam-case",
                coordinate_origin_m=np.zeros(3),
                density_unit="kg/m^3",
                coordinate_unit="m",
                volume_unit="m^3",
                time_unit="s",
                provenance=_provenance(export_commands[time_name]),
            )
            roundtrip = _assert_native_roundtrip(
                snapshot,
                native_points=points.values,
                native_centres=centres.values,
                native_volumes=volumes.values,
                native_density=density.values,
                expected_count=CELL_COUNT,
            )
            snapshots.append(snapshot)

            # From this point onward, every discrete scientific value comes only
            # from the unchanged reader Snapshot.
            weighted = snapshot.as_volume_element_source().as_weighted_source()
            production = {
                "potential": point_potential(weighted, TARGETS_M),
                "acceleration": point_acceleration(weighted, TARGETS_M),
                "gradient": point_gravity_gradient(weighted, TARGETS_M),
            }
            literal = literal_discrete_observables(
                snapshot.positions_m, weighted.weights_kg, TARGETS_M
            )
            continuum = point_anomaly_observables(centre_m(time_s), TARGETS_M)
            scales = _discrete_scales(
                snapshot.positions_m, weighted.weights_kg, TARGETS_M
            )
            literal_checks: dict[str, Any] = {}
            continuum_errors: dict[str, list[float]] = {}
            for observable in production:
                error_norm = _norm_per_target(
                    production[observable] - literal[observable]
                )
                reference_norm = _norm_per_target(literal[observable])
                tolerance = 1e-12 * reference_norm + 1e-14 * scales[observable]
                literal_checks[observable] = {
                    "error_norm": error_norm.tolist(),
                    "reference_norm": reference_norm.tolist(),
                    "scale": scales[observable].tolist(),
                    "tolerance": tolerance.tolist(),
                    "passed": bool(np.all(error_norm <= tolerance)),
                }
                continuum_errors[observable] = (
                    _norm_per_target(production[observable] - continuum[observable])
                    / _norm_per_target(continuum[observable])
                ).tolist()
            mass = float(np.sum(weighted.weights_kg))
            radial = TARGETS_M - centre_m(time_s)
            translation_sign = bool(
                np.all(production["potential"] > 0.0)
                and np.all(
                    np.einsum("mi,mi->m", production["acceleration"], radial) > 0.0
                )
            )
            observations[time_name] = {
                "classification": "observed reader payload and derived calculation",
                "vtk_path": str(vtk_path),
                "vtk_sha256": snapshot.provenance["input_sha256"],
                "time_s": snapshot.time_s,
                "time_name": snapshot.time_name,
                "cell_count": len(snapshot.cell_ids),
                "native_export_audit": roundtrip,
                "native_dtypes": {
                    key: snapshot.provenance[key]
                    for key in (
                        "points_dtype",
                        "centre_dtype",
                        "volume_dtype",
                        "density_dtype",
                        "cell_id_dtype",
                    )
                },
                "reader_payload_hashes": {
                    "cell_ids": hashlib.sha256(snapshot.cell_ids.tobytes()).hexdigest(),
                    "centres": hashlib.sha256(
                        snapshot.positions_m.tobytes()
                    ).hexdigest(),
                    "volumes": hashlib.sha256(
                        snapshot.volumes_m3.tobytes()
                    ).hexdigest(),
                    "density": hashlib.sha256(
                        snapshot.density_kg_m3.tobytes()
                    ).hexdigest(),
                },
                "grid_mass_kg": mass,
                "analytic_mass_kg": analytic_total_mass_kg(),
                "mass_relative_error": abs(mass - analytic_total_mass_kg())
                / abs(analytic_total_mass_kg()),
                "literal_checks": literal_checks,
                "continuum_relative_errors": continuum_errors,
                "translation_sign_passed": translation_sign,
                "production": {
                    key: value.tolist() for key, value in production.items()
                },
                "literal_discrete": {
                    key: value.tolist() for key, value in literal.items()
                },
                "continuum_reference": {
                    "classification": "inferred analytic reference",
                    **{key: value.tolist() for key, value in continuum.items()},
                },
            }

        snapshots[0].assert_compatible_successor(snapshots[1])
        snapshots[1].assert_compatible_successor(snapshots[2])
        shift = int(round(0.5 / (4.0 / LEVEL)))
        density_arrays = [snapshot.density_kg_m3 for snapshot in snapshots]
        translated = all(
            np.array_equal(
                np.roll(
                    density_arrays[index].reshape((LEVEL,) * 3, order="F"),
                    shift,
                    axis=0,
                ),
                density_arrays[index + 1].reshape((LEVEL,) * 3, order="F"),
            )
            for index in (0, 1)
        )
    masses = np.asarray(
        [observations[time_name]["grid_mass_kg"] for time_name in TIMES]
    )
    sum_abs_weights = np.asarray(
        [
            np.sum(
                np.abs(
                    snapshot.as_volume_element_source().as_weighted_source().weights_kg
                )
            )
            for snapshot in snapshots
        ]
    )
    mass_time_tolerance = float(
        128 * np.finfo(np.float32).eps * np.max(sum_abs_weights)
    )
    max_mass_time_difference = float(np.max(np.abs(masses - masses[0])))
    per_time_errors = {
        time_name: observations[time_name]["continuum_relative_errors"]
        for time_name in TIMES
    }
    summaries = {
        observable: summarize_continuum_errors(per_time_errors, observable)
        for observable in ("potential", "acceleration", "gradient")
    }
    return {
        "status": "complete",
        "classification": "observed reader payload and derived calculation",
        "level": LEVEL,
        "analysis_duration_s": time.monotonic() - started,
        "analysis_timeout_s": ANALYSIS_TIMEOUT_S,
        "process_peak_rss_kib_before": process_rss_before,
        "process_peak_rss_kib_after": resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss,
        "science_source_contract": (
            "production/literal/scales/mass/successor/translation/sign checks use "
            "only unchanged load_openfoam_density Snapshot values"
        ),
        "successor_compatibility_passed": True,
        "native_translation_cells": shift,
        "native_translation_passed": translated,
        "max_mass_time_difference_kg": max_mass_time_difference,
        "mass_time_tolerance_kg": mass_time_tolerance,
        "mass_time_consistency_passed": max_mass_time_difference <= mass_time_tolerance,
        "continuum_error_summaries": summaries,
        "times": observations,
    }


def _repository_delivery(output: Path, worktree: Path) -> dict[str, Any]:
    relative_files = (
        Path("scripts/research_openfoam_cosine_bell_precision.py"),
        Path("tests/test_openfoam_cosine_bell_precision.py"),
    )
    delivery = output / "delivery"
    raw_root = delivery / "new-files"
    records: dict[str, Any] = {}
    patch_parts = []
    for relative in relative_files:
        source = worktree / relative
        if subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(relative)],
            cwd=worktree,
            check=False,
            capture_output=True,
        ).returncode == 0:
            raise AssertionError(f"expected base-absent new file: {relative}")
        target = raw_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        diff = subprocess.run(
            ["git", "diff", "--no-index", "--", "/dev/null", str(relative)],
            cwd=worktree,
            check=False,
            capture_output=True,
            text=True,
        )
        if diff.returncode != 1:
            raise RuntimeError(f"could not capture new-file patch for {relative}")
        patch_parts.append(diff.stdout)
        records[str(relative)] = {
            "before": "absent",
            "after_bytes": source.stat().st_size,
            "after_sha256": _sha256(source),
            "raw_copy": str(target),
            "raw_copy_sha256": _sha256(target),
        }
    patch_path = delivery / "repository.patch"
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_text("".join(patch_parts), encoding="utf-8")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=worktree,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != BASE_SHA:
        raise AssertionError(
            f"worktree HEAD {head} differs from approved base {BASE_SHA}"
        )
    return {
        "base_sha": head,
        "files": records,
        "raw_new_file_patch": {
            "path": str(patch_path),
            "bytes": patch_path.stat().st_size,
            "sha256": _sha256(patch_path),
        },
    }


def run_validation(
    prior_run: Path,
    export_manifest_path: Path,
    case: Path,
    output: Path,
    worktree: Path,
) -> None:
    """Run the bounded export audit and 48^3 scientific validation."""
    prior_run = Path(prior_run).resolve()
    export_manifest_path = Path(export_manifest_path).resolve()
    case = Path(case).resolve()
    output = Path(output).resolve()
    worktree = Path(worktree).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    logs = output / "logs"
    logs.mkdir()
    started_at = _iso_now()
    stage = "resource preflight"
    try:
        free_before = shutil.disk_usage(output).free
        if free_before < MIN_FREE_BYTES:
            raise RuntimeError("less than 10 GiB free before scientific validation")
        retained_fixed_inputs = _all_retained_run_bytes(
            prior_run.parent
        ) + _tree_bytes(case)
        retained_before = retained_fixed_inputs + _tree_bytes(output)
        if retained_before > MAX_RETAINED_BYTES:
            raise RuntimeError("prior plus new retained scientific data exceeds 1 GiB")

        stage = "prior evidence hash and derivation verification"
        prior_levels, prior_primary = _verify_prior(prior_run)
        stage = "export manifest and VTK hash verification"
        export_manifest, vtk_outputs, export_commands = _verify_export_manifest(
            export_manifest_path, case
        )
        stage = "native export audit and reader-snapshot scientific calculation"
        level48 = _analyse_48(case, export_commands)
        _write_json(output / "scientific-level-n48.json", level48)

        stage = "eight-gate evaluation"
        maxima = _prior_maxima(prior_levels)
        maxima["mass"]["48"] = max(
            level48["times"][time_name]["mass_relative_error"]
            for time_name in TIMES
        )
        for observable in ("potential", "acceleration", "gradient"):
            maxima[observable]["48"] = level48["continuum_error_summaries"][
                observable
            ]["maximum"]
        gates = evaluate_gates(maxima)
        supporting_checks = {
            "all_three_times_loaded": set(level48["times"]) == set(TIMES),
            "all_literal_checks_passed": all(
                item["literal_checks"][observable]["passed"]
                for item in level48["times"].values()
                for observable in ("potential", "acceleration", "gradient")
            ),
            "successor_compatibility_passed": level48[
                "successor_compatibility_passed"
            ],
            "density_bitwise_translation_passed": level48[
                "native_translation_passed"
            ],
            "mass_time_consistency_passed": level48[
                "mass_time_consistency_passed"
            ],
            "translation_signs_passed": all(
                item["translation_sign_passed"]
                for item in level48["times"].values()
            ),
        }
        derivation = prior_primary["derivation"]
        scientific_status = classify_scientific_status(gates, supporting_checks)
        science_manifest = {
            "status": scientific_status,
            "classifications": {
                "new_export_reader_payloads_and_results": "observed",
                "continuum_reference": "inferred; verified by retained prior evidence",
                "acceptance_thresholds": "fixed by the approved plan before execution",
            },
            "model": {
                "levels": [16, 32, 48],
                "precision_qualification": {
                    "16": "legacy VTK float32, 6 significant-digit ASCII",
                    "32": "legacy VTK float32, 6 significant-digit ASCII",
                    "48": "legacy VTK float32, 9 significant-digit ASCII",
                    "interpretation": (
                        "mixed export precision comparison; not a pure spatial "
                        "mesh-convergence or convergence-order result"
                    ),
                },
                "times_s": list(TIMES_S),
                "targets_m": TARGETS_M.tolist(),
                "field": "delta_rho only; uniform background gravity excluded",
            },
            "prior_reference_evidence": {
                "classification": "retained reference evidence",
                "scientific_manifest_path": str(prior_run / "scientific-manifest.json"),
                "scientific_manifest_sha256": PRIOR_PRIMARY_EVIDENCE_HASH,
                "level_hashes": PRIOR_LEVEL_HASHES,
                "CAS_mass_passed": derivation["mass_exact_passed"],
                "high_precision_quadrature_passed": derivation["quadrature_passed"],
                "advection_passed": derivation["advection_exact_passed"],
                "boundary_passed": derivation["boundary_continuity_passed"],
            },
            "level_48": level48,
            "max_errors": maxima,
            "gates": gates,
            "supporting_checks": supporting_checks,
            "all_scientific_gates_passed": scientific_status == "complete",
        }
        _write_json(output / "scientific-manifest.json", science_manifest)

        stage = "repository patch capture"
        delivery = _repository_delivery(output, worktree)
        retained_after = _retained_science_bytes(prior_run, case, output)
        if retained_after > MAX_RETAINED_BYTES:
            raise RuntimeError("prior plus new retained scientific data exceeds 1 GiB")
        resource_manifest = {
            "status": "complete",
            "classification": "observed",
            "free_bytes_before": free_before,
            "free_bytes_after": shutil.disk_usage(output).free,
            "required_free_bytes": MIN_FREE_BYTES,
            "prior_plus_new_scientific_bytes_before": retained_before,
            "prior_runs_plus_new_case_bytes": retained_fixed_inputs,
            "prior_plus_new_scientific_bytes_before_final_manifests": retained_after,
            "authoritative_post_capture_count": "see runner-manifest.json",
            "retained_limit_bytes": MAX_RETAINED_BYTES,
            "analysis_timeout_s": ANALYSIS_TIMEOUT_S,
            "maximum_snapshots": 3,
            "maximum_cells_per_snapshot": CELL_COUNT,
        }
        _write_json(output / "resource-manifest.json", resource_manifest)
        evidence = {
            "status": scientific_status,
            "classification": "observed unless nested classification says otherwise",
            "started_at": started_at,
            "ended_at": _iso_now(),
            "source_origin": {
                "interpreter": sys.executable,
                "gragra": __import__("gragra").__file__,
                "openfoam_adapter": sys.modules[
                    "gragra.adapters.openfoam_density"
                ].__file__,
                "old_helper_script": sys.modules[
                    "scripts.research_openfoam_cosine_bell"
                ].__file__,
                "study_script": str(Path(__file__).resolve()),
            },
            "logical_worker_invocation": {
                "argv": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
                "qualification": (
                    "logical worker argv only; actual source-launcher process record "
                    "is captured by the outer runner after this worker exits"
                ),
            },
            "export_manifest": {
                "path": str(export_manifest_path),
                "sha256": _sha256(export_manifest_path),
                "schema_version": export_manifest.get("schema_version"),
                "status": export_manifest["status"],
            },
            "vtk_outputs": vtk_outputs,
            "export_qualification": {
                "binary": (
                    "stock Foundation 14 foamToVTK executable with a private, "
                    "patched libfileFormats.so selected for this export"
                ),
                "patch_scope": "legacy VTK ASCII floatScalar precision only",
                "payload": (
                    "float32 values represented with 9 significant ASCII digits; "
                    "not float64 VTK payloads"
                ),
            },
            "repository_delivery": delivery,
            "manifests": {},
            "scientific_gates_passed": science_manifest[
                "all_scientific_gates_passed"
            ],
        }
        for name in (
            "scientific-level-n48.json",
            "scientific-manifest.json",
            "resource-manifest.json",
        ):
            path = output / name
            evidence["manifests"][name] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        _write_json(output / "evidence-manifest.json", evidence)
    except Exception as error:
        failure = {
            "status": "partial",
            "classification": "observed failure",
            "started_at": started_at,
            "ended_at": _iso_now(),
            "failure": {
                "stage": stage,
                "type": type(error).__name__,
                "message": str(error),
            },
            "scientific_result": (
                "unevaluated"
                if stage != "eight-gate evaluation"
                else "see partial outputs"
            ),
            "uncomputed_values": "unevaluated; no values fabricated",
            "logical_worker_invocation": {
                "argv": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
                "qualification": (
                    "logical worker argv only; actual source-launcher process record "
                    "is captured by the outer runner after this worker exits"
                ),
            },
        }
        _write_json(output / "failure-manifest.json", failure)
        raise


def _capture_actual_launcher(prior_run: Path, case: Path, output: Path) -> int:
    """Run one worker through the actual source launcher and capture its process."""
    output = Path(output).resolve()
    prior_run = Path(prior_run).resolve()
    case = Path(case).resolve()
    if output.exists():
        raise FileExistsError(output)
    command = list(sys.orig_argv)
    if len(command) < 2:
        raise RuntimeError("cannot recover the actual Python launcher argv")
    cwd = Path.cwd().resolve()
    environment = os.environ.copy()
    environment["GRAGRA_O48_VALIDATION_WORKER"] = "1"
    started_at = _iso_now()
    timed_out = False
    try:
        process = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            capture_output=True,
            timeout=ANALYSIS_TIMEOUT_S,
            check=False,
        )
        exit_code = process.returncode
        stdout = process.stdout
        stderr = process.stderr
    except subprocess.TimeoutExpired as error:
        timed_out = True
        exit_code = 124
        stdout = error.stdout or b""
        stderr = error.stderr or b""
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / "actual-launcher.stdout.log"
    stderr_path = logs / "actual-launcher.stderr.log"
    stdout_path.write_bytes(stdout)
    stderr_path.write_bytes(stderr)
    record = {
        "status": "complete" if exit_code == 0 else "partial",
        "classification": "observed subprocess execution",
        "actual_launcher_argv": command,
        "cwd": str(cwd),
        "started_at": started_at,
        "ended_at": _iso_now(),
        "timeout_s": ANALYSIS_TIMEOUT_S,
        "timed_out": timed_out,
        "source_launcher_worker_exit": exit_code,
        "exit": exit_code,
        "stdout": {
            "path": str(stdout_path),
            "bytes": stdout_path.stat().st_size,
            "sha256": _sha256(stdout_path),
        },
        "stderr": {
            "path": str(stderr_path),
            "bytes": stderr_path.stat().st_size,
            "sha256": _sha256(stderr_path),
        },
    }
    runner_path = output / "runner-manifest.json"
    _write_json(runner_path, record)
    result_path = output / (
        "evidence-manifest.json" if exit_code == 0 else "failure-manifest.json"
    )
    if result_path.is_file():
        result = _json(result_path)
        result["actual_launcher_capture"] = {
            "path": str(runner_path),
            "bytes": runner_path.stat().st_size,
            "sha256": _sha256(runner_path),
        }
        _write_json(result_path, result)
    elif exit_code == 0:
        exit_code = 1
        record["status"] = "partial"
        record["exit"] = exit_code
        record["failure"] = "worker exited zero without evidence-manifest.json"
        _write_json(runner_path, record)
    for _iteration in range(4):
        retained = _retained_science_bytes(prior_run, case, output)
        previous = record.get("authoritative_post_capture_retained_bytes")
        record["authoritative_post_capture_retained_bytes"] = retained
        record["retained_limit_bytes"] = MAX_RETAINED_BYTES
        record["retained_limit_passed"] = retained <= MAX_RETAINED_BYTES
        _write_json(runner_path, record)
        if result_path.is_file():
            result = _json(result_path)
            result["actual_launcher_capture"] = {
                "path": str(runner_path),
                "bytes": runner_path.stat().st_size,
                "sha256": _sha256(runner_path),
            }
            _write_json(result_path, result)
        if previous == retained:
            break
    if retained > MAX_RETAINED_BYTES:
        exit_code = 1
        record["status"] = "partial"
        record["exit"] = exit_code
        record["failure"] = {
            "stage": "post-capture retained-byte limit",
            "message": "prior O runs plus new case and science outputs exceed 1 GiB",
        }
        _write_json(runner_path, record)
        if result_path.is_file():
            result = _json(result_path)
            result["status"] = "partial"
            result["outer_runner_failure"] = record["failure"]
            result["actual_launcher_capture"] = {
                "path": str(runner_path),
                "bytes": runner_path.stat().st_size,
                "sha256": _sha256(runner_path),
            }
            _write_json(result_path, result)
    return exit_code


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-run", required=True, type=Path)
    parser.add_argument("--export-manifest", required=True, type=Path)
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--worktree", required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get("GRAGRA_O48_VALIDATION_WORKER") == "1":
        run_validation(
            args.prior_run,
            args.export_manifest,
            args.case,
            args.output_dir,
            args.worktree,
        )
    else:
        raise SystemExit(
            _capture_actual_launcher(args.prior_run, args.case, args.output_dir)
        )


if __name__ == "__main__":
    main()
