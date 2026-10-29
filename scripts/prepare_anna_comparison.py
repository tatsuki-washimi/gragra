"""Prepare a neutral tetrahedral input and comparison contract for ANNA.

This module deliberately has no ANNA reader or execution path.  It produces
gragra reference outputs and a labelled synthetic candidate for exercising the
metadata/normalization checks only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

import gragra
from gragra.constants import G_SI
from gragra.fields import UnstructuredDisplacementField, piecewise_tetra_linear_sources
from gragra.observables import displacement_transfer_acceleration, point_acceleration
from gragra.waves import plane_wave_displacement

EVIDENCE_ROOT = Path(
    "/data/gragra-research/2026-09-22-analytic-followup/20260921T180520Z"
)
RHO_KG_M3 = 2700.0
U_REF_M = 1.0e-6
T_REF_S = 0.0
COORDINATE_ORIGIN_M = np.zeros(3, dtype=np.float64)
WAVE_PHASE0 = 0.37
K_HAT = np.array([1.0, 2.0, 2.0], dtype=np.float64) / 3.0
WAVENUMBER_RAD_M = np.pi / 2.0
S_POLARIZATION = np.array([2.0, -1.0, 0.0], dtype=np.float64) / np.sqrt(5.0)
ACCELERATION_SCALE = G_SI * RHO_KG_M3 * U_REF_M
INPUT_UNITS = {
    "coordinate": "m",
    "displacement": "m",
    "density": "kg/m^3",
    "time": "s",
    "wavenumber": "rad/m",
}
OUTPUT_UNITS = "m/s^2"
COMPONENT_ORDER = ["x", "y", "z"]
WAVE_PHASE_CONVENTION = (
    "exp(+i*(wavenumber_rad_m * (k_hat dot (x-coordinate_origin_m)) + wave_phase0_rad))"
)
CONTRACT_VERSION = "anna-neutral-tetra4-v1"
FIELD_CONTRACT = {
    "interior_representation": "P1_nodal_interpolation",
    "nodal_values_are_the_input": True,
    "continuous_wave_comparison_is_separate": True,
    "discrete_s_bulk_is_not_assumed_zero": True,
    "discrete_divergence_units": "1",
}
GRAGRA_QUADRATURE_CONTRACT = {
    "mass_positions": "tetra_centroid",
    "mass_displacement": "tetra_nodal_mean",
    "bulk_source": "tetra_centroid_weighted_point_mass",
    "interface_source": "boundary_face_centroid_weighted_surface_mass",
    "surface_displacement": "boundary_face_nodal_mean",
    "exact_fe_volume_integral": False,
    "interpretation": (
        "Existing gragra centroid quadrature reference; not an exact FE volume "
        "integral."
    ),
}
FUTURE_QUADRATURE_BUDGET = [
    "gragra tetra/face centroid quadrature error",
    "unknown ANNA integration rule and its convergence error",
    "difference between the two quadrature rules",
]
CASE_CONTRACTS = {
    "constant": {
        "output_name": "acceleration",
        "observable": "acceleration",
        "units": OUTPUT_UNITS,
        "component_order": COMPONENT_ORDER,
        "expected_dtype": "float64",
        "coordinate_origin_m": [0.0, 0.0, 0.0],
        "wave_phase0_rad": None,
        "phase_convention": "not_applicable_for_nonwave_case",
        "wave_type": None,
        "wavenumber_rad_m": None,
        "k_hat": None,
        "polarization": None,
    },
    "affine": {
        "output_name": "acceleration",
        "observable": "acceleration",
        "units": OUTPUT_UNITS,
        "component_order": COMPONENT_ORDER,
        "expected_dtype": "float64",
        "coordinate_origin_m": [0.0, 0.0, 0.0],
        "wave_phase0_rad": None,
        "phase_convention": "not_applicable_for_nonwave_case",
        "wave_type": None,
        "wavenumber_rad_m": None,
        "k_hat": None,
        "polarization": None,
    },
    "complex_p": {
        "output_name": "acceleration",
        "observable": "acceleration",
        "units": OUTPUT_UNITS,
        "component_order": COMPONENT_ORDER,
        "expected_dtype": "complex128",
        "coordinate_origin_m": [0.0, 0.0, 0.0],
        "wave_phase0_rad": WAVE_PHASE0,
        "phase_convention": WAVE_PHASE_CONVENTION,
        "wave_type": "P",
        "wavenumber_rad_m": WAVENUMBER_RAD_M,
        "k_hat": K_HAT.tolist(),
        "polarization": K_HAT.tolist(),
    },
    "complex_s": {
        "output_name": "acceleration",
        "observable": "acceleration",
        "units": OUTPUT_UNITS,
        "component_order": COMPONENT_ORDER,
        "expected_dtype": "complex128",
        "coordinate_origin_m": [0.0, 0.0, 0.0],
        "wave_phase0_rad": WAVE_PHASE0,
        "phase_convention": WAVE_PHASE_CONVENTION,
        "wave_type": "custom",
        "wavenumber_rad_m": WAVENUMBER_RAD_M,
        "k_hat": K_HAT.tolist(),
        "polarization": S_POLARIZATION.tolist(),
    },
}

NODES_M = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ],
    dtype=np.float64,
)
TETRA = np.array(
    [
        [0, 1, 2, 6],
        [0, 2, 3, 6],
        [0, 3, 7, 6],
        [0, 7, 4, 6],
        [0, 4, 5, 6],
        [0, 5, 1, 6],
    ],
    dtype=np.int64,
)
BOUNDARY_FACES = np.array(
    [
        [0, 3, 2],
        [0, 2, 1],
        [4, 5, 6],
        [4, 6, 7],
        [0, 7, 3],
        [0, 4, 7],
        [1, 2, 6],
        [1, 6, 5],
        [0, 1, 5],
        [0, 5, 4],
        [3, 7, 6],
        [3, 6, 2],
    ],
    dtype=np.int64,
)
TARGETS_M = np.array(
    [[2.0, 0.25, 0.50], [-0.30, 1.80, 0.70], [0.40, -0.60, 2.20]],
    dtype=np.float64,
)
AFFINE_OFFSET = np.array([1.0, -0.5, 0.25], dtype=np.float64)
AFFINE_MATRIX = np.array(
    [[0.2, 0.1, -0.3], [0.4, -0.1, 0.2], [0.0, 0.3, 0.5]],
    dtype=np.float64,
)


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array).copy()
    result.setflags(write=False)
    return result


def _complex_json(value: Any) -> Any:
    if isinstance(value, complex):
        return [float(value.real), float(value.imag)]
    if isinstance(value, np.generic):
        return _complex_json(value.item())
    if isinstance(value, np.ndarray):
        return _complex_json(value.tolist())
    if isinstance(value, list):
        return [_complex_json(item) for item in value]
    if isinstance(value, tuple):
        return [_complex_json(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _complex_json(item) for key, item in value.items()}
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Return the stable UTF-8 representation used for input hashes."""

    return json.dumps(
        _complex_json(value),
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def array_hash(array: np.ndarray) -> str:
    """Hash dtype, shape, and C-order bytes without relying on NPZ metadata."""

    value = np.ascontiguousarray(array)
    header = f"{value.dtype.str}|{value.shape}".encode("ascii")
    return sha256_bytes(header + b"\0" + value.tobytes(order="C"))


def boundary_normals() -> np.ndarray:
    points = NODES_M[BOUNDARY_FACES]
    cross = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    return _readonly(cross / np.linalg.norm(cross, axis=1)[:, None])


def displacement_cases() -> dict[str, np.ndarray]:
    """Return fixed nodal displacement cases, including complex P/S cases."""

    constant = np.broadcast_to(U_REF_M * AFFINE_OFFSET, NODES_M.shape).copy()
    affine = U_REF_M * (AFFINE_OFFSET[None, :] + (NODES_M @ AFFINE_MATRIX.T))
    wave_positions = NODES_M - COORDINATE_ORIGIN_M
    p_wave = plane_wave_displacement(
        wave_positions,
        [K_HAT],
        [WAVENUMBER_RAD_M],
        "P",
        amplitude_m=U_REF_M,
        phase0=WAVE_PHASE0,
    )[:, 0, 0, :]
    s_wave = plane_wave_displacement(
        wave_positions,
        [K_HAT],
        [WAVENUMBER_RAD_M],
        "custom",
        amplitude_m=U_REF_M,
        phase0=WAVE_PHASE0,
        polarization=S_POLARIZATION,
    )[:, 0, 0, :]
    return {
        "constant": _readonly(constant),
        "affine": _readonly(affine),
        "complex_p": _readonly(p_wave),
        "complex_s": _readonly(s_wave),
    }


def fixture_metadata(cases: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "field_contract": FIELD_CONTRACT,
        "gragra_quadrature_contract": GRAGRA_QUADRATURE_CONTRACT,
        "quadrature_comparison_status": "NOT_ASSUMED_IDENTICAL",
        "future_quadrature_budget": FUTURE_QUADRATURE_BUDGET,
        "input_hash": input_hash(cases),
        "target_order_hash": target_order_hash(),
        "input_units": INPUT_UNITS,
        "G_SI": G_SI,
        "normalization_scale_m_s2": ACCELERATION_SCALE,
        "target_count": len(TARGETS_M),
        "coordinate_origin_m": COORDINATE_ORIGIN_M.tolist(),
        "t_ref_s": T_REF_S,
    }


def case_metadata(cases: dict[str, np.ndarray], case_id: str) -> dict[str, Any]:
    try:
        contract = CASE_CONTRACTS[case_id]
    except KeyError as error:
        raise ValueError(f"unknown fixture case {case_id!r}") from error
    return {
        **fixture_metadata(cases),
        **contract,
        "case_id": case_id,
        "case_nodal_hash": array_hash(cases[case_id]),
    }


def input_payload(cases: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "field_contract": FIELD_CONTRACT,
        "gragra_quadrature_contract": GRAGRA_QUADRATURE_CONTRACT,
        "future_quadrature_budget": FUTURE_QUADRATURE_BUDGET,
        "case_contracts": CASE_CONTRACTS,
        "nodes_m": NODES_M,
        "tetra": TETRA,
        "boundary_faces": BOUNDARY_FACES,
        "targets_m": TARGETS_M,
        "rho_kg_m3": RHO_KG_M3,
        "G_SI": G_SI,
        "input_units": INPUT_UNITS,
        "coordinate_origin_m": COORDINATE_ORIGIN_M,
        "t_ref_s": T_REF_S,
        "u_ref_m": U_REF_M,
        "wave_phase0_rad": WAVE_PHASE0,
        "k_hat": K_HAT,
        "wavenumber_rad_m": WAVENUMBER_RAD_M,
        "s_polarization": S_POLARIZATION,
        "cases": cases,
    }


def input_hash(cases: dict[str, np.ndarray]) -> str:
    return sha256_bytes(canonical_json_bytes(input_payload(cases)))


def target_order_hash() -> str:
    return array_hash(TARGETS_M)


def comparison_metadata(cases: dict[str, np.ndarray], case_id: str) -> dict[str, Any]:
    """Return the complete metadata contract for one fixture case."""

    return case_metadata(cases, case_id)


def build_results(
    cases: dict[str, np.ndarray],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    """Run only gragra's NumPy tetra/observable path and retain raw SI arrays."""

    results: dict[str, dict[str, np.ndarray]] = {}
    divergences: dict[str, np.ndarray] = {}
    for name, displacement in cases.items():
        field = UnstructuredDisplacementField(NODES_M, TETRA, displacement)
        sources = piecewise_tetra_linear_sources(field, RHO_KG_M3)
        bulk_density = np.asarray(sources.bulk_source.density_kg_m3)
        divergence = _readonly(-bulk_density / RHO_KG_M3)
        total = displacement_transfer_acceleration(
            sources.mass_positions_m,
            sources.mass_kg,
            sources.mass_displacement_m,
            TARGETS_M,
            backend="numpy",
        )
        bulk = point_acceleration(
            sources.bulk_source.as_weighted_source(), TARGETS_M, backend="numpy"
        )
        interface = point_acceleration(
            sources.interface_source, TARGETS_M, backend="numpy"
        )
        surface = sources.interface_field.as_surface_mass_source(RHO_KG_M3)
        results[name] = {
            "displacement_m": _readonly(displacement),
            "divergence": divergence,
            "bulk_density_kg_m3": _readonly(bulk_density),
            "bulk_weights_kg": _readonly(
                sources.bulk_source.as_weighted_source().weights_kg
            ),
            "bulk_acceleration_si": _readonly(bulk),
            "interface_acceleration_si": _readonly(interface),
            "total_acceleration_si": _readonly(total),
            "normalized_acceleration": _readonly(total / ACCELERATION_SCALE),
            "interface_triangles": _readonly(sources.interface_field.triangles),
            "interface_normals": _readonly(surface.normals),
            "interface_areas_m2": _readonly(surface.areas_m2),
            "interface_normal_displacement_m": _readonly(
                np.asarray(surface.normal_displacement_m)
            ),
            "interface_weights_kg": _readonly(sources.interface_source.weights_kg),
        }
        divergences[name] = divergence
    return results, divergences


def _metadata_mismatch(expected: dict[str, Any], actual: dict[str, Any]) -> str:
    for key in (
        "contract_version",
        "input_hash",
        "case_nodal_hash",
        "target_order_hash",
        "case_id",
        "observable",
        "output_name",
        "units",
        "component_order",
        "expected_dtype",
        "target_count",
        "coordinate_origin_m",
        "wave_phase0_rad",
        "phase_convention",
        "wave_type",
        "wavenumber_rad_m",
        "k_hat",
        "polarization",
        "G_SI",
        "t_ref_s",
        "normalization_scale_m_s2",
    ):
        if actual.get(key) != expected.get(key):
            return f"metadata mismatch for {key!r}"
    return ""


def compare_acceleration_outputs(
    reference_si: np.ndarray,
    candidate_si: np.ndarray,
    candidate_metadata: dict[str, Any],
    expected_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Compare two raw-SI arrays using the fixed common normalization only."""

    mismatch = _metadata_mismatch(expected_metadata, candidate_metadata)
    if mismatch:
        raise ValueError(mismatch)
    reference = np.asarray(reference_si)
    candidate = np.asarray(candidate_si)
    if reference.shape != candidate.shape:
        raise ValueError(
            f"target-order/shape mismatch: {candidate.shape} != {reference.shape}"
        )
    target_count = expected_metadata["target_count"]
    if reference.shape != (target_count, len(COMPONENT_ORDER)):
        raise ValueError(
            "target_count/shape mismatch: "
            f"{reference.shape} != ({target_count}, {len(COMPONENT_ORDER)})"
        )
    expected_dtype = np.dtype(expected_metadata["expected_dtype"]).name
    if reference.dtype.name != expected_dtype:
        raise ValueError(
            f"dtype mismatch for reference: {reference.dtype.name} != {expected_dtype}"
        )
    if candidate.dtype.name != expected_dtype:
        raise ValueError(
            f"dtype mismatch for candidate: {candidate.dtype.name} != {expected_dtype}"
        )
    if reference.ndim != 2 or reference.shape[1] != len(COMPONENT_ORDER):
        raise ValueError("acceleration arrays must have shape (target_count, 3)")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("acceleration arrays must be finite")
    scale = float(expected_metadata["normalization_scale_m_s2"])
    if scale <= 0.0 or not np.isfinite(scale):
        raise ValueError("normalization scale must be positive and finite")
    delta_si = candidate - reference
    delta_normalized = delta_si / scale
    return {
        "candidate_source": candidate_metadata.get("candidate_source", "unknown"),
        "normalization_scale_m_s2": scale,
        "max_abs_delta_si": float(np.max(np.abs(delta_si))),
        "max_abs_delta_normalized": float(np.max(np.abs(delta_normalized))),
        "reference_acceleration_si": reference,
        "candidate_acceleration_si": candidate,
        "reference_acceleration_normalized": reference / scale,
        "candidate_acceleration_normalized": candidate / scale,
        "delta_acceleration_si": delta_si,
        "delta_acceleration_normalized": delta_normalized,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _result_npz_arrays(
    results: dict[str, dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "nodes_m": NODES_M,
        "tetra": TETRA,
        "boundary_faces": BOUNDARY_FACES,
        "boundary_normals": boundary_normals(),
        "targets_m": TARGETS_M,
    }
    for case_name, case_result in results.items():
        prefix = f"{case_name}_"
        arrays.update({prefix + key: value for key, value in case_result.items()})
    return arrays


def _manifest_dict(manifest: Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(manifest, Path):
        return json.loads(manifest.read_text())
    return manifest


def read_saved_data(
    npz_path: Path, manifest: Path | dict[str, Any]
) -> dict[str, np.ndarray]:
    """Read every saved array and verify its manifest dtype, shape, and hash."""

    manifest_data = _manifest_dict(manifest)
    expected_metadata = manifest_data.get("array_metadata")
    if not isinstance(expected_metadata, dict):
        raise ValueError("manifest lacks array_metadata")
    expected_keys = set(expected_metadata)
    expected_hashes = manifest_data.get("array_hashes", {})
    arrays: dict[str, np.ndarray] = {}
    with np.load(npz_path, allow_pickle=False) as exported:
        actual_keys = set(exported.files)
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        if missing:
            raise ValueError(f"missing NPZ keys: {missing}")
        if unexpected:
            raise ValueError(f"unexpected NPZ keys: {unexpected}")
        for key in sorted(expected_keys):
            array = np.asarray(exported[key])
            saved = expected_metadata[key]
            if array.dtype.str != saved["dtype"]:
                raise ValueError(
                    f"dtype mismatch for {key}: {array.dtype.str} != {saved['dtype']}"
                )
            if list(array.shape) != saved["shape"]:
                raise ValueError(
                    f"shape mismatch for {key}: {list(array.shape)} != {saved['shape']}"
                )
            actual_hash = array_hash(array)
            if actual_hash != saved["sha256"]:
                raise ValueError(f"hash mismatch for {key}")
            if expected_hashes.get(key) != actual_hash:
                raise ValueError(f"hash mismatch for {key} in array_hashes")
            arrays[key] = array.copy()
    return arrays


def read_and_verify_input_contract(
    input_path: Path, manifest: Path | dict[str, Any]
) -> dict[str, Any]:
    """Reload the canonical input JSON and recompute its manifest hash."""

    manifest_data = _manifest_dict(manifest)
    payload = json.loads(input_path.read_text())
    actual_hash = sha256_bytes(canonical_json_bytes(payload))
    if actual_hash != manifest_data["input_hash"]:
        raise ValueError(
            f"input JSON hash mismatch: {actual_hash} != {manifest_data['input_hash']}"
        )
    return payload


def write_evidence(evidence_root: Path = EVIDENCE_ROOT) -> dict[str, Path]:
    """Generate external preparation artifacts and return their paths."""

    results_dir = evidence_root / "results" / "anna"
    logs_dir = evidence_root / "logs" / "anna"
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    cases = displacement_cases()
    metadata = fixture_metadata(cases)
    case_metadata_map = {
        case_id: comparison_metadata(cases, case_id) for case_id in cases
    }
    results, divergences = build_results(cases)
    arrays = _result_npz_arrays(results)
    npz_path = results_dir / "anna_gragra_results.npz"
    np.savez(npz_path, **arrays)

    array_metadata = {
        name: {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "sha256": array_hash(value),
        }
        for name, value in arrays.items()
    }
    array_hashes = {name: saved["sha256"] for name, saved in array_metadata.items()}
    manifest = {
        **metadata,
        "case_contracts": CASE_CONTRACTS,
        "case_metadata": case_metadata_map,
        "state": ["PREPARATION_COMPLETE", "ANNA_EXECUTION_NOT_RUN"],
        "candidate_source": "gragra_numpy_reference",
        "gragra_file": str(Path(gragra.__file__).resolve()),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "argv": list(sys.argv),
        "array_metadata": array_metadata,
        "array_hashes": array_hashes,
        "raw_result_npz": str(npz_path),
    }
    manifest_path = results_dir / "anna_gragra_manifest.json"
    _write_json(manifest_path, manifest)
    input_path = results_dir / "anna_input_contract.json"
    _write_json(input_path, input_payload(cases))
    hash_path = results_dir / "anna_hashes.json"
    _write_json(
        hash_path,
        {
            "input_hash": metadata["input_hash"],
            "target_order_hash": metadata["target_order_hash"],
            "array_metadata": array_metadata,
            "array_hashes": array_hashes,
        },
    )
    verified_arrays = read_saved_data(npz_path, manifest)
    verified_input = read_and_verify_input_contract(input_path, manifest)
    readback_path = results_dir / "anna_hash_readback.json"
    recomputed_array_hashes = {
        key: array_hash(value) for key, value in verified_arrays.items()
    }
    recomputed_input_hash = sha256_bytes(canonical_json_bytes(verified_input))

    synthetic_scale = 1.25
    synthetic_case = "complex_p"
    synthetic_key = "candidate_acceleration_si"
    synthetic_candidate = (
        results[synthetic_case]["total_acceleration_si"] * synthetic_scale
    )
    synthetic_array_metadata = {
        synthetic_key: {
            "dtype": synthetic_candidate.dtype.str,
            "shape": list(synthetic_candidate.shape),
            "sha256": array_hash(synthetic_candidate),
        }
    }
    synthetic_array_hashes = {
        synthetic_key: synthetic_array_metadata[synthetic_key]["sha256"]
    }
    synthetic_metadata = {
        **case_metadata_map[synthetic_case],
        "candidate_source": "synthetic_test_double",
        "provenance": "controlled amplitude mismatch for plumbing tests only",
        "npz_key": synthetic_key,
        "array_metadata": synthetic_array_metadata,
        "array_hashes": synthetic_array_hashes,
    }
    synthetic_path = results_dir / "anna_synthetic_test_double.npz"
    np.savez(synthetic_path, **{synthetic_key: synthetic_candidate})
    synthetic_metadata_path = results_dir / "anna_synthetic_test_double.json"
    _write_json(synthetic_metadata_path, synthetic_metadata)
    verified_synthetic = read_saved_data(synthetic_path, synthetic_metadata)
    synthetic_recomputed_hashes = {
        key: array_hash(value) for key, value in verified_synthetic.items()
    }
    synthetic_comparison = compare_acceleration_outputs(
        results[synthetic_case]["total_acceleration_si"],
        verified_synthetic[synthetic_key],
        synthetic_metadata,
        case_metadata_map[synthetic_case],
    )
    synthetic_comparison.update(
        {
            "candidate_loaded_from_saved_npz": True,
            "candidate_npz_path": str(synthetic_path),
            "candidate_npz_key": synthetic_key,
        }
    )
    comparison_path = results_dir / "anna_synthetic_comparison.json"
    _write_json(comparison_path, synthetic_comparison)

    _write_json(
        readback_path,
        {
            "npz_path": str(npz_path),
            "manifest_path": str(manifest_path),
            "array_count": len(verified_arrays),
            "keys": sorted(verified_arrays),
            "all_array_hashes_match": recomputed_array_hashes == array_hashes,
            "array_metadata": manifest["array_metadata"],
            "recomputed_array_hashes": recomputed_array_hashes,
            "input_json_hash": manifest["input_hash"],
            "input_json_hash_recomputed": recomputed_input_hash,
            "input_json_hash_match": recomputed_input_hash == manifest["input_hash"],
            "synthetic": {
                "npz_path": str(synthetic_path),
                "metadata_path": str(synthetic_metadata_path),
                "array_count": len(verified_synthetic),
                "keys": sorted(verified_synthetic),
                "all_array_hashes_match": synthetic_recomputed_hashes
                == synthetic_array_hashes,
                "array_metadata": synthetic_array_metadata,
                "recomputed_array_hashes": synthetic_recomputed_hashes,
            },
        },
    )

    validation_path = logs_dir / "prepare_anna_comparison.json"
    _write_json(
        validation_path,
        {
            "state": manifest["state"],
            "python": sys.executable,
            "gragra_file": manifest["gragra_file"],
            "argv": list(sys.argv),
            "case_count": len(cases),
            "tetra_count": len(TETRA),
            "boundary_face_count": len(BOUNDARY_FACES),
            "targets": len(TARGETS_M),
            "input_hash": metadata["input_hash"],
            "divergence_units": FIELD_CONTRACT["discrete_divergence_units"],
            "case_metadata": case_metadata_map,
            "synthetic_max_abs_delta_normalized": synthetic_comparison[
                "max_abs_delta_normalized"
            ],
            "limitations": [
                "ANNA execution not run",
                "native ANNA format not inferred",
                "synthetic candidate is not ANNA output",
                "P2-3 deferred",
            ],
            "divergence_by_case": divergences,
        },
    )
    return {
        "manifest": manifest_path,
        "input": input_path,
        "hashes": hash_path,
        "readback": readback_path,
        "results": npz_path,
        "synthetic": synthetic_path,
        "synthetic_metadata": synthetic_metadata_path,
        "comparison": comparison_path,
        "validation": validation_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=EVIDENCE_ROOT,
        help="external evidence root",
    )
    args = parser.parse_args()
    paths = write_evidence(args.evidence_root)
    for name, path in paths.items():
        print(f"{name}: {path}")
    print("state: PREPARATION_COMPLETE + ANNA_EXECUTION_NOT_RUN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
