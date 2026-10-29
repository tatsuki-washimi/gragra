from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from test_specfem_reader import _write_custom_time_sampled_gll

import gragra
from gragra.adapters.specfem import schema
from gragra.observables.dipole import displacement_transfer_acceleration
from gragra.waves.rayleigh import rayleigh_displacement
from scripts import research_wave6_specfem_analysis as analysis

ROOT = Path(__file__).resolve().parents[1]


def test_imports_lane_source() -> None:
    assert Path(gragra.__file__).resolve() == ROOT / "src/gragra/__init__.py"


def _element_coordinates(origins: np.ndarray) -> np.ndarray:
    local = np.linspace(0.0, 0.8, 5, dtype=np.float64)
    x, y, z = np.meshgrid(local, local, local, indexing="ij")
    offsets = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    return origins[:, None, :] + offsets[None, :, :]


def _write_fixture(
    path: Path,
    origins: np.ndarray,
    *,
    samples: int,
    actual_dt_s: float,
    phasor: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coordinates = _element_coordinates(origins)
    elements = len(origins)
    density = np.full((elements, 125), 2300.0, dtype=np.float64)
    volume = np.linspace(0.8, 1.2, elements * 125, dtype=np.float64).reshape(
        elements, 125
    )
    if phasor is None:
        phasor = rayleigh_displacement(
            coordinates.reshape(-1, 3),
            analysis.DIRECTION,
            analysis.WAVENUMBER_RAD_M,
            analysis.VP_M_S,
            analysis.VS_M_S,
            amplitude_m=analysis.AMPLITUDE_M,
        ).reshape(elements, 125, 3)
    native_steps = np.arange(1, samples + 1, dtype=np.float64)
    phase = analysis.OMEGA_RAD_S * native_steps * actual_dt_s
    displacement = np.real(
        phasor[None, :, :, :] * np.exp(-1j * phase)[:, None, None, None]
    )
    _write_custom_time_sampled_gll(
        path,
        ngll_shape=(5, 5, 5),
        coordinates=coordinates,
        density=density,
        quadrature_volume=volume,
        displacement=displacement,
    )
    return coordinates, density, volume, phasor


def test_two_window_fit_uses_native_physical_time_rule() -> None:
    rng = np.random.default_rng(20260912)
    expected = rng.normal(size=(7, 3)) + 1j * rng.normal(size=(7, 3))
    native_steps = np.arange(1, 241, dtype=np.int64)
    actual_dt_s = np.float64(0.004999999888241291)
    theta = analysis.OMEGA_RAD_S * native_steps.astype(np.float64) * actual_dt_s
    values = np.real(expected[None, :, :] * np.exp(-1j * theta)[:, None, None])

    fitted = analysis.fit_two_windows(
        values,
        native_steps,
        actual_dt_s=actual_dt_s,
        window_ends=(160, 240),
    )

    np.testing.assert_allclose(fitted[160], expected, rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(fitted[240], expected, rtol=2e-15, atol=2e-15)


def test_case_analysis_uses_stream_reader_and_512_element_units(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    origins = np.array(
        [[0.0, 0.0, -4.0], [2.0, 0.0, -4.0], [4.0, 0.0, -4.0]],
        dtype=np.float64,
    )
    phasor = rng.normal(scale=1e-9, size=(3, 125, 3)) + 1j * rng.normal(
        scale=1e-9, size=(3, 125, 3)
    )
    path = tmp_path / "case.h5"
    coordinates, density, volume, expected_phasor = _write_fixture(
        path, origins, samples=4, actual_dt_s=0.01, phasor=phasor
    )
    targets = np.array([[20.0, 20.0, 10.0], [25.0, -10.0, 8.0]])

    result = analysis.analyze_case(
        path,
        length_m=12_000,
        actual_dt_s=0.01,
        targets_m=targets,
        fixed_c=np.zeros((2, 3), dtype=np.complex128),
        fixed_norm_c=np.ones(2),
        window_ends=(3, 4),
        source_block_elements=2,
    )

    mass = (density * volume).reshape(-1)
    expected_a = displacement_transfer_acceleration(
        coordinates.reshape(-1, 3),
        mass,
        expected_phasor.reshape(-1, 3),
        targets,
        chunk_size=analysis.KERNEL_CHUNK_SOURCES,
    )
    expected_b_phasor = rayleigh_displacement(
        coordinates.reshape(-1, 3),
        analysis.DIRECTION,
        analysis.WAVENUMBER_RAD_M,
        analysis.VP_M_S,
        analysis.VS_M_S,
        amplitude_m=analysis.AMPLITUDE_M,
    )
    expected_b = displacement_transfer_acceleration(
        coordinates.reshape(-1, 3),
        mass,
        expected_b_phasor,
        targets,
        chunk_size=analysis.KERNEL_CHUNK_SOURCES,
    )
    assert result["source_count"] == 3 * 125
    assert result["processed_element_count"] == 3
    assert result["block_count"] == 2
    assert result["source_block_elements"] == 2
    assert result["reader_class"] == "SpecfemGLLStreamReader"
    for end in (3, 4):
        np.testing.assert_allclose(result["windows"][end]["A"], expected_a, rtol=5e-14)
        np.testing.assert_allclose(result["windows"][end]["B"], expected_b, rtol=2e-15)
        assert result["windows"][end]["A"].dtype == np.complex128
    assert result["analytic_fit_diagnostic"]["components_checked"] == 3


def test_nested_overlap_is_bitwise_and_reaggregates_element_contributions(
    tmp_path: Path,
) -> None:
    h5py = pytest.importorskip("h5py")
    inner_origins = np.array([[0.0, 0.0, -4.0], [2.0, 0.0, -4.0]], dtype=np.float64)
    outer_origins = np.array(
        [
            [-2.0, 0.0, -4.0],
            [0.0, 0.0, -4.0],
            [2.0, 0.0, -4.0],
            [4.0, 0.0, -4.0],
        ],
        dtype=np.float64,
    )
    inner = tmp_path / "inner.h5"
    outer = tmp_path / "outer.h5"
    _write_fixture(inner, inner_origins, samples=3, actual_dt_s=0.01)
    _write_fixture(outer, outer_origins, samples=3, actual_dt_s=0.01)
    # Make overlap density/volume byte-identical despite the fixture's
    # different total element counts.
    with h5py.File(inner, "r") as source, h5py.File(outer, "r+") as target:
        target[schema.PATH_MATERIAL_DENSITY][1:3] = source[
            schema.PATH_MATERIAL_DENSITY
        ][:]
        target[schema.PATH_MESH_QUADRATURE_VOLUME][1:3] = source[
            schema.PATH_MESH_QUADRATURE_VOLUME
        ][:]

    exact = analysis.compare_nested_overlap(inner, outer, block_elements=2)
    assert exact["status"] == "NESTED_OVERLAP_BITWISE_ACCEPTED"
    assert exact["overlap_element_count"] == 2
    assert exact["missing_key_count"] == 0
    assert exact["duplicate_key_count"] == 0
    assert exact["extra_overlap_key_count"] == 0
    assert exact["all_static_fields_bitwise_equal"] is True
    assert exact["b_element_contributions_bitwise_equal"] is True
    assert exact["b_element_contributions_status"] == "COMPLETE"
    assert exact["b_element_contributions_evaluated_element_count"] == 2

    with h5py.File(outer, "r+") as target:
        target[schema.PATH_MESH_COORDINATES][1, 0, 0] = np.float64(-0.0)
    signed_zero = analysis.compare_nested_overlap(inner, outer, block_elements=2)
    assert signed_zero["status"] == "NESTED_OVERLAP_BITWISE_REJECTED"
    assert signed_zero["exact_by_field"]["coordinates"] is False
    assert signed_zero["max_abs_coordinate_difference"] == 0.0
    assert signed_zero["first_mismatch_key"] == [0.0, 0.0, -4.0]
    assert signed_zero["domain_size_difference_interpretable"] is False
    assert signed_zero["b_element_contributions_bitwise_equal"] is False
    assert signed_zero["b_element_contributions_status"] == "NOT_EVALUATED"
    assert signed_zero["b_element_contributions_evaluated_element_count"] == 0
    assert signed_zero["b_element_contribution_inner_sha256"] is None
    assert signed_zero["b_overlap_inner_same_order"] is None

    with h5py.File(outer, "r+") as target:
        target[schema.PATH_MESH_COORDINATES][1, 0, 0] = np.float64(0.0)
        value = target[schema.PATH_MESH_QUADRATURE_VOLUME][1, 17]
        target[schema.PATH_MESH_QUADRATURE_VOLUME][1, 17] = value + 1.0e-6
    changed = analysis.compare_nested_overlap(inner, outer, block_elements=2)
    assert changed["status"] == "NESTED_OVERLAP_BITWISE_REJECTED"
    assert changed["exact_by_field"]["quadrature_volume"] is False
    assert changed["domain_size_difference_interpretable"] is False
    assert changed["first_mismatch_key"] == [0.0, 0.0, -4.0]
    assert changed["max_abs_quadrature_volume_difference"] > 0.0
    assert changed["b_element_contributions_bitwise_equal"] is False
    assert changed["b_element_contributions_status"] == "NOT_EVALUATED"
    assert changed["b_element_contributions_evaluated_element_count"] == 0
    assert changed["b_overlap_inner_same_order"] is None
    np.testing.assert_array_equal(
        exact["b_overlap_inner_same_order"], exact["b_overlap_outer_same_order"]
    )

    with h5py.File(outer, "r+") as changed:
        dataset = changed[schema.PATH_MESH_QUADRATURE_VOLUME]
        dataset[1, 7] = float(dataset[1, 7]) + 1e-12
    mismatch = analysis.compare_nested_overlap(inner, outer, block_elements=2)
    assert mismatch["status"] == "NESTED_OVERLAP_BITWISE_REJECTED"
    assert mismatch["all_static_fields_bitwise_equal"] is False
    assert mismatch["domain_size_difference_interpretable"] is False
    assert mismatch["first_mismatch_key"] == [0.0, 0.0, -4.0]
    assert mismatch["max_abs_quadrature_volume_difference"] > 0.0


def _complete_case_result(length_m: int) -> dict:
    source_count = (
        (48 * 48 * 12) if length_m == 12_000 else (72 * 72 * 12)
    ) * analysis.NGLL_PER_ELEMENT
    zeros = np.zeros_like(analysis.FIXED_C_M_S2)
    windows = {
        str(stop): analysis._jsonable(
            {"A": zeros, "B": zeros, "C": analysis.FIXED_C_M_S2}
        )
        for stop in analysis.WINDOW_ENDS
    }
    return {
        "status": "CASE_ANALYSIS_COMPLETE",
        "full_case": True,
        "length_m": length_m,
        "actual_dt_s": 0.00499999989,
        "source_count": source_count,
        "processed_source_count": source_count,
        "rss_budget_passed": True,
        "rss_budget_bytes": analysis.MEMORY_BUDGET_BYTES,
        "process_peak_rss_bytes_cumulative": analysis.MEMORY_BUDGET_BYTES,
        "window_ends_inclusive": list(analysis.WINDOW_ENDS),
        "windows": windows,
        "target_labels": list(analysis.TARGET_LABELS),
        "targets_m": analysis.TARGETS_M.tolist(),
        "fixed_norm_C_m_s2": analysis.FIXED_NORM_C_M_S2.tolist(),
    }


def test_pair_comparison_rejects_representative_and_wrong_case_results() -> None:
    case_12 = _complete_case_result(12_000)
    case_18 = _complete_case_result(18_000)
    overlap = {"domain_size_difference_interpretable": True}

    accepted = analysis.pair_comparison(case_12, case_18, overlap)
    assert accepted["status"] == "PAIR_COMPARISON_COMPLETE"

    representative = dict(case_12, status="REPRESENTATIVE_BLOCK_COMPLETE")
    with pytest.raises(ValueError, match="not a complete full-case"):
        analysis.pair_comparison(representative, case_18, overlap)

    swapped = dict(case_12, length_m=18_000)
    with pytest.raises(ValueError, match="expected L12000"):
        analysis.pair_comparison(swapped, case_18, overlap)

    partial = dict(case_12, processed_source_count=case_12["source_count"] - 125)
    with pytest.raises(ValueError, match="partial source coverage"):
        analysis.pair_comparison(partial, case_18, overlap)


def test_pair_comparison_rejects_target_window_and_reference_mismatch() -> None:
    case_18 = _complete_case_result(18_000)
    overlap = {"domain_size_difference_interpretable": True}

    wrong_target = _complete_case_result(12_000)
    wrong_target["targets_m"][0][0] += 1.0
    with pytest.raises(ValueError, match="target coordinates"):
        analysis.pair_comparison(wrong_target, case_18, overlap)

    missing_window = _complete_case_result(12_000)
    missing_window["windows"].pop("160")
    with pytest.raises(ValueError, match="window set is incomplete"):
        analysis.pair_comparison(missing_window, case_18, overlap)

    wrong_c = _complete_case_result(12_000)
    wrong_c["windows"]["160"]["C"][0][0][0] += 1.0
    with pytest.raises(ValueError, match="window 160 C does not match"):
        analysis.pair_comparison(wrong_c, case_18, overlap)

    wrong_norm = _complete_case_result(12_000)
    wrong_norm["fixed_norm_C_m_s2"][0] *= 2.0
    with pytest.raises(ValueError, match=r"norm\(C\)"):
        analysis.pair_comparison(wrong_norm, case_18, overlap)


@pytest.mark.parametrize(
    "bad_peak",
    [
        pytest.param(True, id="bool"),
        pytest.param(-1, id="negative"),
        pytest.param(1.0, id="float"),
        pytest.param("1", id="string"),
        pytest.param(None, id="missing"),
    ],
)
def test_pair_comparison_rejects_malformed_final_peak_rss(bad_peak) -> None:
    case_12 = _complete_case_result(12_000)
    case_18 = _complete_case_result(18_000)
    if bad_peak is None:
        case_12.pop("process_peak_rss_bytes_cumulative")
    else:
        case_12["process_peak_rss_bytes_cumulative"] = bad_peak

    with pytest.raises(ValueError, match="passing final RSS check"):
        analysis.pair_comparison(
            case_12, case_18, {"domain_size_difference_interpretable": True}
        )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        pytest.param("rss_budget_passed", 1, id="pass-flag-not-bool"),
        pytest.param("rss_budget_passed", False, id="pass-flag-false"),
        pytest.param("rss_budget_bytes", 2**31 - 1, id="wrong-budget"),
        pytest.param("rss_budget_bytes", float(2**31), id="budget-float"),
        pytest.param("rss_budget_bytes", None, id="budget-missing"),
    ],
)
def test_pair_comparison_requires_exact_final_rss_budget_record(
    field: str, bad_value
) -> None:
    case_12 = _complete_case_result(12_000)
    if bad_value is None:
        case_12.pop(field)
    else:
        case_12[field] = bad_value

    with pytest.raises(ValueError, match="passing final RSS check"):
        analysis.pair_comparison(
            case_12,
            _complete_case_result(18_000),
            {"domain_size_difference_interpretable": True},
        )


def test_json_publication_rejects_nan_and_retains_failure(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    with pytest.raises(ValueError, match="Out of range float"):
        analysis._write_json_no_overwrite(output, {"bad": np.nan})

    assert not output.exists()
    assert output.with_suffix(".json.partial.failure.json").exists()


def test_json_publication_rejects_finite_readback_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "result.json"
    partial = output.with_suffix(".json.partial")
    original_read_text = Path.read_text

    def changed_read_text(path: Path, *args, **kwargs) -> str:
        observed = original_read_text(path, *args, **kwargs)
        return (
            observed.replace('"value": 1', '"value": 2')
            if path == partial
            else observed
        )

    monkeypatch.setattr(Path, "read_text", changed_read_text)
    with pytest.raises(ValueError, match="reopened readback mismatch"):
        analysis._write_json_no_overwrite(output, {"value": 1})

    assert not output.exists()
    assert partial.exists()
    assert output.with_suffix(".json.partial.failure.json").exists()


def test_full_analysis_requires_matching_accepted_rss_preflight() -> None:
    report = {
        "status": "RSS_PREFLIGHT_ACCEPTED",
        "length_m": 12_000,
        "source_block_elements": 512,
        "processed_element_count": 512,
        "rss_budget_passed": True,
        "rss_budget_bytes": analysis.MEMORY_BUDGET_BYTES,
        "process_peak_rss_bytes_cumulative": analysis.MEMORY_BUDGET_BYTES,
    }
    analysis._validate_rss_preflight(report, length_m=12_000, source_block_elements=512)

    with pytest.raises(ValueError, match="accepted RSS preflight"):
        analysis._validate_rss_preflight(
            dict(report, status="RSS_PREFLIGHT_REJECTED_SHRINK_BLOCK_AND_RETRY"),
            length_m=12_000,
            source_block_elements=512,
        )
    with pytest.raises(ValueError, match="block size"):
        analysis._validate_rss_preflight(
            report, length_m=12_000, source_block_elements=256
        )


@pytest.mark.parametrize(
    "bad_peak",
    [
        pytest.param(True, id="bool"),
        pytest.param(-1, id="negative"),
        pytest.param(1.0, id="float"),
        pytest.param("1", id="string"),
        pytest.param(None, id="missing"),
    ],
)
def test_rss_preflight_rejects_malformed_peak_rss(bad_peak) -> None:
    report = {
        "status": "RSS_PREFLIGHT_ACCEPTED",
        "length_m": 12_000,
        "source_block_elements": 512,
        "processed_element_count": 512,
        "rss_budget_passed": True,
        "rss_budget_bytes": analysis.MEMORY_BUDGET_BYTES,
        "process_peak_rss_bytes_cumulative": analysis.MEMORY_BUDGET_BYTES,
    }
    if bad_peak is None:
        report.pop("process_peak_rss_bytes_cumulative")
    else:
        report["process_peak_rss_bytes_cumulative"] = bad_peak

    with pytest.raises(ValueError, match="passing 2 GiB budget record"):
        analysis._validate_rss_preflight(
            report, length_m=12_000, source_block_elements=512
        )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        pytest.param("rss_budget_passed", 1, id="pass-flag-not-bool"),
        pytest.param("rss_budget_passed", False, id="pass-flag-false"),
        pytest.param("rss_budget_bytes", 2**31 - 1, id="wrong-budget"),
        pytest.param("rss_budget_bytes", float(2**31), id="budget-float"),
        pytest.param("rss_budget_bytes", None, id="budget-missing"),
    ],
)
def test_rss_preflight_requires_exact_budget_record(field: str, bad_value) -> None:
    report = {
        "status": "RSS_PREFLIGHT_ACCEPTED",
        "length_m": 12_000,
        "source_block_elements": 512,
        "processed_element_count": 512,
        "rss_budget_passed": True,
        "rss_budget_bytes": analysis.MEMORY_BUDGET_BYTES,
        "process_peak_rss_bytes_cumulative": analysis.MEMORY_BUDGET_BYTES,
    }
    if bad_value is None:
        report.pop(field)
    else:
        report[field] = bad_value

    with pytest.raises(ValueError, match="passing 2 GiB budget record"):
        analysis._validate_rss_preflight(
            report, length_m=12_000, source_block_elements=512
        )


def test_cli_failure_retains_exit_argv_source_and_utc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "analysis.json"

    def fail_dt(_runs_root: Path) -> float:
        raise ValueError("focused CLI failure")

    monkeypatch.setattr(analysis, "_matched_actual_dt", fail_dt)
    argv = [
        "rss-preflight",
        "12000",
        "--output-json",
        str(output),
        "--canonical-path",
        str(tmp_path / "absent.h5"),
    ]
    with pytest.raises(ValueError, match="focused CLI failure"):
        analysis.main(argv)

    failure = output.with_suffix(".json.execution.failure.json")
    report = json.loads(failure.read_text())
    assert report["status"] == "ANALYSIS_EXECUTION_FAILED_RETAINED"
    assert report["execution"]["exit_status"] == 1
    assert report["execution"]["argv"][1:] == argv
    assert report["execution"]["source"]["sha256"]
    assert report["execution"]["started_utc"].endswith("+00:00")
    assert report["execution"]["completed_utc"].endswith("+00:00")
