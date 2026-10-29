"""Tests for gragra.adapters.specfem.schema.

Covers: contract-value cross-checks for constants/enums, the required-key
tables (base + conditional + oneof), canonical JSON determinism, GLL
flatten/unflatten round-trips, hash formatting, and (locally only) the
2-stage upstream constants.h equivalence check.
"""

import os
import re
from pathlib import Path

import numpy as np
import pytest

from gragra.adapters.specfem import schema

# ===========================================================================
# Constants / enum contract values
# ===========================================================================


def test_schema_identity_constants():
    assert schema.SCHEMA_VERSION == 1
    assert schema.MESH_TYPE == "gll"
    assert schema.LAYOUT_SINGLE_FILE == "single_file"
    assert schema.VALID_OUTPUT_SAMPLING_KINDS == (
        "time_sampled",
        "static",
        "frequency_domain",
    )


def test_idomain_constants():
    assert schema.IDOMAIN_ACOUSTIC == 1
    assert schema.IDOMAIN_ELASTIC == 2
    assert schema.IDOMAIN_POROELASTIC == 3
    assert {1, 2, 3} == schema.VALID_DOMAIN_IDS


def test_element_id_methods():
    assert schema.ELEMENT_ID_METHODS == (
        "pre_partition_global_index",
        "explicit_glob2loc_map",
        "corner_node_tuple",
    )
    assert schema.ADOPTED_ELEMENT_ID_METHOD == "pre_partition_global_index"
    assert schema.ADOPTED_ELEMENT_ID_METHOD in schema.ELEMENT_ID_METHODS


def test_upstream_constant_expectations():
    assert schema.EXPECTED_NGLL == (5, 5, 5)
    assert schema.EXPECTED_GAUSSALPHA == 0.0
    assert schema.EXPECTED_GAUSSBETA == 0.0
    assert schema.EXPECTED_USE_MESH_COLORING_GPU is False


# ===========================================================================
# Dataset spec table
# ===========================================================================


def test_dataset_specs_cover_schema_a():
    expected_paths = {
        schema.PATH_MESH_COORDINATES,
        schema.PATH_MESH_QUADRATURE_VOLUME,
        schema.PATH_MESH_ELEMENT_ID,
        schema.PATH_MESH_PARTITION_ID,
        schema.PATH_MESH_DOMAIN_ID,
        schema.PATH_MESH_MATERIAL_ID,
        schema.PATH_MESH_IS_PML,
        schema.PATH_MESH_ELEMENT_CORNER_NODES,
        schema.PATH_MATERIAL_DENSITY,
        schema.PATH_FIELD_DISPLACEMENT,
        schema.PATH_TIME,
        schema.PATH_TIME_STEP,
        schema.PATH_DIAGNOSTICS_DIVERGENCE,
        schema.PATH_DIAGNOSTICS_CURL,
    }
    assert set(schema.DATASET_SPECS.keys()) == expected_paths


def test_dataset_spec_dtypes_and_shapes():
    coords = schema.DATASET_SPECS[schema.PATH_MESH_COORDINATES]
    assert coords.dtype == np.dtype(np.float64)
    assert coords.shape == ("element", "gll", "3")
    assert coords.output_modes == {"static", "time_sampled"}

    is_pml = schema.DATASET_SPECS[schema.PATH_MESH_IS_PML]
    assert is_pml.dtype == np.dtype(np.bool_)
    assert is_pml.required is True

    element_id = schema.DATASET_SPECS[schema.PATH_MESH_ELEMENT_ID]
    assert element_id.dtype == np.dtype(np.int64)

    partition_id = schema.DATASET_SPECS[schema.PATH_MESH_PARTITION_ID]
    assert partition_id.dtype == np.dtype(np.int32)

    displacement = schema.DATASET_SPECS[schema.PATH_FIELD_DISPLACEMENT]
    assert displacement.output_modes == {"time_sampled"}
    assert schema.PATH_TIME not in {
        p
        for p, spec in schema.DATASET_SPECS.items()
        if "static" in spec.output_modes and p == schema.PATH_TIME
    }
    time_spec = schema.DATASET_SPECS[schema.PATH_TIME]
    assert time_spec.output_modes == {"time_sampled"}


def test_element_corner_nodes_required():
    assert schema.element_corner_nodes_required("corner_node_tuple") is True
    assert schema.element_corner_nodes_required("pre_partition_global_index") is False
    assert schema.element_corner_nodes_required("explicit_glob2loc_map") is False


def test_required_dataset_paths_modes():
    static_paths = schema.required_dataset_paths("static")
    time_sampled_paths = schema.required_dataset_paths("time_sampled")
    assert schema.PATH_FIELD_DISPLACEMENT not in static_paths
    assert schema.PATH_TIME not in static_paths
    assert schema.PATH_TIME_STEP not in static_paths
    assert schema.PATH_FIELD_DISPLACEMENT in time_sampled_paths
    assert schema.PATH_TIME in time_sampled_paths
    assert schema.PATH_MESH_COORDINATES in static_paths
    assert schema.PATH_MESH_COORDINATES in time_sampled_paths
    # element_corner_nodes is conditional, never in the base required set.
    assert schema.PATH_MESH_ELEMENT_CORNER_NODES not in static_paths
    assert schema.PATH_MESH_ELEMENT_CORNER_NODES not in time_sampled_paths


# ===========================================================================
# Provenance required-key tables
# ===========================================================================


def test_provenance_key_groups_are_disjoint():
    common = set(schema.COMMON_PROVENANCE_KEYS)
    specfem = set(schema.SPECFEM_PROVENANCE_KEYS)
    cadence = set(schema.CADENCE_PROVENANCE_KEYS)
    assert common & specfem == set()
    assert common & cadence == set()
    assert specfem & cadence == set()
    assert set(schema.ALL_PROVENANCE_KEYS) == common | specfem | cadence


def test_required_keys_rejects_invalid_output_sampling_kind():
    with pytest.raises(ValueError):
        schema.required_keys("bogus")


def test_required_keys_static_excludes_cadence_and_time_axis_keys():
    required = schema.required_keys("static")
    for name in schema.CADENCE_PROVENANCE_KEYS:
        assert name not in required, f"{name} must not be required for static output"
    time_axis_keys = (
        "t0",
        "t0_source",
        "dt_s",
        "nstep",
        "time_step_origin",
        "native_step_origin",
    )
    for name in time_axis_keys:
        assert name not in required, f"{name} must not be required for static output"
    # Always-required keys must still be present.
    for name in ("schema_version", "mesh_type", "layout", "output_sampling_kind"):
        assert name in required


def test_required_keys_time_sampled_includes_cadence_and_time_axis_keys():
    required = schema.required_keys("time_sampled")
    for name in (
        "output_interval_steps",
        "f_max_analysis",
        "f_min_analysis",
        "minimum_cycles",
        "sampling_oversampling_factor",
        "antialias_filter_applied",
        "cadence_provenance",
        "t0",
        "t0_source",
        "dt_s",
        "nstep",
        "time_step_origin",
        "native_step_origin",
    ):
        assert name in required, f"{name} must be required for time_sampled output"
    # Value-conditional cadence keys are NOT part of the unconditional base set.
    for name in (
        "antialias_filter_spec",
        "f_content_max",
        "f_content_max_method",
        "f_content_max_evidence",
        "alias_error_budget",
    ):
        assert name not in required


def test_required_keys_frequency_domain_is_common_only():
    required = schema.required_keys("frequency_domain")
    assert "schema_version" in required
    assert "t0" not in required
    assert "output_interval_steps" not in required


def test_recommended_keys():
    # t0_crosscheck_max_deviation is recommended alongside t0_crosscheck.
    assert schema.recommended_keys("time_sampled") == {
        "t0_crosscheck",
        "t0_crosscheck_max_deviation",
    }
    assert schema.recommended_keys("static") == frozenset()
    with pytest.raises(ValueError):
        schema.recommended_keys("bogus")


def test_conditionally_required_keys_source_build():
    extra = schema.conditionally_required_keys({"build_kind": "source"})
    assert "specfem_git_commit" in extra
    assert "source_tree_clean" in extra
    assert "package_version" not in extra


def test_conditionally_required_keys_package_build():
    extra = schema.conditionally_required_keys({"build_kind": "package"})
    assert "package_version" in extra
    assert "package_digest" in extra
    assert "specfem_git_commit" not in extra


def test_conditionally_required_keys_antialias_filter():
    extra = schema.conditionally_required_keys({"antialias_filter_applied": True})
    assert "antialias_filter_spec" in extra
    extra_off = schema.conditionally_required_keys({"antialias_filter_applied": False})
    assert "antialias_filter_spec" not in extra_off


def test_conditionally_required_keys_f_content_max():
    extra_m1 = schema.conditionally_required_keys({"output_interval_steps": 1})
    assert "f_content_max" not in extra_m1
    extra_m2 = schema.conditionally_required_keys({"output_interval_steps": 2})
    assert "f_content_max" in extra_m2
    assert "f_content_max_method" in extra_m2


def test_conditionally_required_keys_alias_budget_and_evidence():
    extra_nyquist = schema.conditionally_required_keys(
        {"f_content_max_method": "solver_step_nyquist"}
    )
    assert "alias_error_budget" not in extra_nyquist
    assert "f_content_max_evidence" not in extra_nyquist

    extra_measured = schema.conditionally_required_keys(
        {"f_content_max_method": "measured_spectrum"}
    )
    assert "alias_error_budget" in extra_measured
    assert "f_content_max_evidence" in extra_measured

    extra_filter = schema.conditionally_required_keys(
        {"f_content_max_method": "filter_cutoff"}
    )
    assert "alias_error_budget" in extra_filter
    assert "f_content_max_evidence" not in extra_filter


def test_source_tree_dirty_hash_satisfied():
    # Not a source build: always satisfied.
    assert schema.source_tree_dirty_hash_satisfied({"build_kind": "package"}) is True
    # Source build, clean tree: satisfied regardless of hash keys.
    assert (
        schema.source_tree_dirty_hash_satisfied(
            {"build_kind": "source", "source_tree_clean": True}
        )
        is True
    )
    # Source build, dirty tree, no hash: NOT satisfied.
    assert (
        schema.source_tree_dirty_hash_satisfied(
            {"build_kind": "source", "source_tree_clean": False}
        )
        is False
    )
    # Source build, dirty tree, patch_hash present: satisfied.
    assert (
        schema.source_tree_dirty_hash_satisfied(
            {
                "build_kind": "source",
                "source_tree_clean": False,
                "patch_hash": "sha256:abc",
            }
        )
        is True
    )
    # Source build, dirty tree, source_tree_hash present: satisfied.
    assert (
        schema.source_tree_dirty_hash_satisfied(
            {
                "build_kind": "source",
                "source_tree_clean": False,
                "source_tree_hash": "sha256:abc",
            }
        )
        is True
    )


# ===========================================================================
# Encoding table / value type checks
# ===========================================================================


def test_provenance_encoding_lookup():
    assert schema.provenance_encoding("schema_version") == "scalar_attr"
    assert schema.provenance_encoding("ngll_shape") == "scalar_attr"
    assert schema.provenance_encoding("executables") == "json_dataset"
    assert schema.provenance_encoding("par_file_text") == "text_dataset"
    with pytest.raises(KeyError):
        schema.provenance_encoding("not_a_real_key")


def test_check_value_type():
    assert schema.check_value_type("nproc", 4) is True
    assert schema.check_value_type("nproc", True) is False  # bool != int
    assert schema.check_value_type("dt_s", 0.05) is True
    assert schema.check_value_type("mesh_type", "gll") is True
    assert schema.check_value_type("pml_exclusion_applied", True) is True
    assert schema.check_value_type("ngll_shape", np.array([5, 5, 5])) is True
    assert schema.check_value_type("ngll_shape", np.array([5, 5])) is False
    assert schema.check_value_type("build_configuration", {"anything": 1}) is True
    with pytest.raises(KeyError):
        schema.check_value_type("not_a_real_key", 1)


# ===========================================================================
# Canonical JSON
# ===========================================================================


def test_canonical_json_is_deterministic_regardless_of_key_order():
    a = schema.canonical_json({"b": 1, "a": 2})
    b = schema.canonical_json({"a": 2, "b": 1})
    assert a == b
    assert a == '{"a":2,"b":1}'


def test_canonical_json_rejects_nan_and_inf():
    with pytest.raises(ValueError):
        schema.canonical_json({"x": float("nan")})
    with pytest.raises(ValueError):
        schema.canonical_json({"x": float("inf")})


def test_canonical_json_preserves_non_ascii():
    text = schema.canonical_json({"note": "日本語"})
    assert "日本語" in text


# ===========================================================================
# GLL flatten convention
# ===========================================================================


def test_ngll_from_shape():
    assert schema.ngll_from_shape((5, 5, 5)) == 125
    assert schema.ngll_from_shape((3, 4, 5)) == 60


def test_reshape_shape_zyx():
    assert schema.reshape_shape_zyx((5, 5, 5)) == (5, 5, 5)
    assert schema.reshape_shape_zyx((3, 4, 5)) == (5, 4, 3)


@pytest.mark.parametrize("ngll_shape", [(5, 5, 5), (3, 4, 5), (2, 2, 2)])
def test_flatten_unflatten_round_trip(ngll_shape):
    nx, ny, nz = ngll_shape
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                flat = schema.flatten_index(i, j, k, ngll_shape)
                assert 0 <= flat < nx * ny * nz
                assert schema.unflatten_index(flat, ngll_shape) == (i, j, k)


def test_flatten_index_x_is_innermost():
    # Per the flatten convention, incrementing i by 1 (x, innermost) must
    # advance the flat index by exactly 1.
    ngll_shape = (5, 5, 5)
    base = schema.flatten_index(0, 0, 0, ngll_shape)
    next_i = schema.flatten_index(1, 0, 0, ngll_shape)
    assert next_i - base == 1
    # Incrementing k (z, outermost) must advance by ny*nx.
    next_k = schema.flatten_index(0, 0, 1, ngll_shape)
    assert next_k - base == 5 * 5


def test_flatten_index_out_of_range_raises():
    with pytest.raises(ValueError):
        schema.flatten_index(5, 0, 0, (5, 5, 5))
    with pytest.raises(ValueError):
        schema.unflatten_index(125, (5, 5, 5))
    with pytest.raises(ValueError):
        schema.unflatten_index(-1, (5, 5, 5))


def test_validate_ngll_shape_rejects_nonpositive():
    with pytest.raises(ValueError):
        schema.flatten_index(0, 0, 0, (0, 5, 5))
    with pytest.raises(ValueError):
        schema.flatten_index(0, 0, 0, (5, 5))


# ===========================================================================
# Hash formatting
# ===========================================================================


def test_format_and_parse_hash_round_trip():
    formatted = schema.format_hash("abcd1234")
    assert formatted == "sha256:abcd1234"
    algorithm, hexdigest = schema.parse_hash(formatted)
    assert algorithm == "sha256"
    assert hexdigest == "abcd1234"


def test_format_hash_custom_algorithm():
    assert schema.format_hash("deadbeef", algorithm="sha1") == "sha1:deadbeef"


def test_format_hash_rejects_empty_hexdigest():
    with pytest.raises(ValueError):
        schema.format_hash("")


def test_parse_hash_rejects_malformed():
    with pytest.raises(ValueError):
        schema.parse_hash("no-colon-here")
    with pytest.raises(ValueError):
        schema.parse_hash(":missing-algorithm")
    with pytest.raises(ValueError):
        schema.parse_hash("sha256:")


def test_is_hash_string():
    assert schema.is_hash_string("sha256:abcd") is True
    assert schema.is_hash_string("not-a-hash") is False
    assert schema.is_hash_string(12345) is False


# ===========================================================================
# Upstream constants.h equivalence (2-stage: skip unless
# GRAGRA_SPECFEM_SOURCE is set)
# ===========================================================================


def _find_constants_header(source_root: Path) -> Path:
    candidate = source_root / "setup" / "constants.h"
    if candidate.is_file():
        return candidate
    candidate_in = source_root / "setup" / "constants.h.in"
    if candidate_in.is_file():
        return candidate_in
    pytest.fail(
        f"Neither setup/constants.h nor setup/constants.h.in found under {source_root}"
    )
    raise AssertionError("unreachable")  # pragma: no cover


def _fortran_real_to_float(token: str) -> float:
    """Convert a Fortran double-precision literal (e.g. '0.d0') to float."""
    normalized = token.strip().replace("D", "e").replace("d", "e")
    return float(normalized)


def _extract_one(pattern: str, content: str, label: str) -> str:
    matches = re.findall(pattern, content, flags=re.IGNORECASE | re.MULTILINE)
    if len(matches) != 1:
        pytest.fail(
            f"Expected exactly one match for {label} in constants header, "
            f"found {len(matches)}: pattern={pattern!r}"
        )
    return matches[0]


def test_upstream_constants_equivalence():
    """Cross-check EXPECTED_NGLL/GAUSSALPHA/GAUSSBETA/USE_MESH_COLORING_GPU/
    IDOMAIN_* against the real SPECFEM3D source tree's setup/constants.h (or
    constants.h.in fallback).

    2-stage design: skipped by default (CI / no upstream
    source tree available); runs for real when GRAGRA_SPECFEM_SOURCE points
    at a checked-out SPECFEM3D Cartesian source tree (optional local check).
    """
    source_env = os.environ.get("GRAGRA_SPECFEM_SOURCE")
    if not source_env:
        pytest.skip(
            "GRAGRA_SPECFEM_SOURCE is not set; skipping upstream "
            "constants.h equivalence check (optional local check)"
        )
    source_root = Path(source_env)
    header_path = _find_constants_header(source_root)
    content = header_path.read_text(encoding="utf-8", errors="replace")

    ngllx_str = _extract_one(
        r"integer\s*,\s*parameter\s*::\s*NGLLX\s*=\s*(\d+)", content, "NGLLX"
    )
    ngllx = int(ngllx_str)

    def _resolve_ngll_axis(varname: str) -> int:
        raw = _extract_one(
            rf"integer\s*,\s*parameter\s*::\s*{varname}\s*=\s*(\w+)",
            content,
            varname,
        )
        if raw.isdigit():
            return int(raw)
        if raw.upper() == "NGLLX":
            return ngllx
        pytest.fail(f"Could not resolve {varname} = {raw!r} to an integer")
        raise AssertionError("unreachable")  # pragma: no cover

    nglly = _resolve_ngll_axis("NGLLY")
    ngllz = _resolve_ngll_axis("NGLLZ")
    assert (ngllx, nglly, ngllz) == schema.EXPECTED_NGLL

    gaussalpha_str = _extract_one(
        r"GAUSSALPHA\s*=\s*([0-9.dDeE+-]+)", content, "GAUSSALPHA"
    )
    gaussbeta_str = _extract_one(
        r"GAUSSBETA\s*=\s*([0-9.dDeE+-]+)", content, "GAUSSBETA"
    )
    assert _fortran_real_to_float(gaussalpha_str) == schema.EXPECTED_GAUSSALPHA
    assert _fortran_real_to_float(gaussbeta_str) == schema.EXPECTED_GAUSSBETA

    coloring_str = _extract_one(
        r"logical\s*,\s*parameter\s*::\s*USE_MESH_COLORING_GPU\s*=\s*\.(true|false)\.",
        content,
        "USE_MESH_COLORING_GPU",
    )
    assert (coloring_str.lower() == "true") == schema.EXPECTED_USE_MESH_COLORING_GPU

    idomain_acoustic = int(
        _extract_one(
            r"integer\s*,\s*parameter\s*::\s*IDOMAIN_ACOUSTIC\s*=\s*(\d+)",
            content,
            "IDOMAIN_ACOUSTIC",
        )
    )
    idomain_elastic = int(
        _extract_one(
            r"integer\s*,\s*parameter\s*::\s*IDOMAIN_ELASTIC\s*=\s*(\d+)",
            content,
            "IDOMAIN_ELASTIC",
        )
    )
    idomain_poroelastic = int(
        _extract_one(
            r"integer\s*,\s*parameter\s*::\s*IDOMAIN_POROELASTIC\s*=\s*(\d+)",
            content,
            "IDOMAIN_POROELASTIC",
        )
    )
    assert idomain_acoustic == schema.IDOMAIN_ACOUSTIC
    assert idomain_elastic == schema.IDOMAIN_ELASTIC
    assert idomain_poroelastic == schema.IDOMAIN_POROELASTIC
