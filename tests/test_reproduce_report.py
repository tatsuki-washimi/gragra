"""Regression tests for reproduction report formatting and evaluation logic."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from scripts.reproduce_analytic_stage1_report import (
    evaluate_status,
    format_json_report,
    format_markdown_tables,
    format_text_tables,
)


def _make_dummy_row(
    shape: str = "直方体 (Cube)",
    condition: str = "内部点 x=[0.2, 0.3, 0.4]",
    observable: str = "a_x",
    unit: str = "m/s^2 (G-free)",
    method: str = "表面求積 (G-free)",
    quad_cond: str = "epsabs=1e-12, epsrel=1e-12",
    scale: float = 3282.53895753319,
    scale_str: str = "3.2825389575331900e+03",
    analytic: float = -3282.53895753319,
    analytic_str: str = "-3.2825389575331897e+03",
    reference: float = -3282.53895753319,
    reference_str: str = "-3.2825389575331919e+03",
    abs_diff: float = 2.2737367544323206e-12,
    abs_diff_str: str = "2.2737367544323206e-12",
    rel_err: float = 6.926762e-16,
    rel_err_str: str = "6.93e-16",
    threshold: float = 1e-11,
    threshold_str: str = "1.00e-11",
    status: str = "PASS",
) -> dict[str, Any]:
    return {
        "shape": shape,
        "condition": condition,
        "observable": observable,
        "unit": unit,
        "method": method,
        "quad_cond": quad_cond,
        "scale": scale,
        "scale_str": scale_str,
        "analytic": analytic,
        "analytic_str": analytic_str,
        "reference": reference,
        "reference_str": reference_str,
        "abs_diff": abs_diff,
        "abs_diff_str": abs_diff_str,
        "rel_err": rel_err,
        "rel_err_str": rel_err_str,
        "threshold": threshold,
        "threshold_str": threshold_str,
        "status": status,
    }


def _make_dummy_row_4_2() -> dict[str, Any]:
    return {
        "case": "Case A (低周波)",
        "condition": "a=1.0, R=5.0, k=0.001, rho=2500.0, xi=1e-06",
        "unit": "kg/m^2",
        "analytic": 1.23456789012345e-04,
        "analytic_str": "+1.23456789012345e-04",
        "epsrel": "1e-08",
        "quadrature": 1.23456789012340e-04,
        "quadrature_str": "+1.23456789012340e-04",
        "abs_err": 5.0e-18,
        "abs_err_str": "5.00e-18",
        "rel_err": 4.05e-14,
        "rel_err_str": "4.05e-14",
    }


def test_evaluate_status_all_pass() -> None:
    rows = [_make_dummy_row(observable=f"obs_{i}") for i in range(26)]
    exit_code, summary_msg = evaluate_status(rows)
    assert exit_code == 0
    assert "SUCCESS" in summary_msg
    assert "All 26/26 verification items in §4.1 passed successfully!" in summary_msg
    assert "§4.2 は観測記録のため判定対象外" in summary_msg


def test_evaluate_status_with_failure() -> None:
    rows = [_make_dummy_row(observable=f"obs_{i}") for i in range(25)]
    fail_row = _make_dummy_row(observable="obs_fail", status="FAIL")
    rows.append(fail_row)

    exit_code, summary_msg = evaluate_status(rows)
    assert exit_code == 1
    assert "FAILURE" in summary_msg
    assert "1/26 verification item(s) failed in §4.1!" in summary_msg
    assert "SUCCESS" not in summary_msg
    assert "passed successfully" not in summary_msg


def test_format_markdown_tables_columns() -> None:
    rows_4_1 = [_make_dummy_row()]
    rows_4_2 = [_make_dummy_row_4_2()]
    md = format_markdown_tables(rows_4_1, rows_4_2)

    required_headers = [
        "対象形状",
        "評価点 / 条件",
        "検証量",
        "単位",
        "解析値 (Analytic)",
        "独立参照値 (Reference)",
        "規格化分母 (Scale)",
        "絶対差 (Abs Diff)",
        "規格化相対誤差 (Rel Err)",
        "受入閾値",
        "判定",
    ]
    for h in required_headers:
        assert h in md, f"Missing header in markdown table: {h}"

    assert "-3.2825389575331897e+03" in md
    assert "-3.2825389575331919e+03" in md
    assert "2.2737367544323206e-12" in md
    assert "3.2825389575331900e+03" in md
    assert "PASS" in md
    assert "観測記録" in md


def test_format_text_tables_columns() -> None:
    rows_4_1 = [_make_dummy_row()]
    rows_4_2 = [_make_dummy_row_4_2()]
    text = format_text_tables(rows_4_1, rows_4_2)

    for h in ["対象形状", "評価条件", "検証量", "単位", "解析値", "独立参照値", "実測相対誤差", "判定"]:
        assert h in text
    assert "-3.2825389575331897e+03" in text
    assert "-3.2825389575331919e+03" in text
    assert "PASS" in text


def test_format_json_report() -> None:
    rows_4_1 = [_make_dummy_row()]
    rows_4_2 = [_make_dummy_row_4_2()]
    json_str = format_json_report(rows_4_1, rows_4_2)
    data = json.loads(json_str)

    assert "summary" in data
    assert "table_4_1" in data
    assert "table_4_2_convergence" in data
    assert data["summary"]["total_items"] == 1
    assert data["summary"]["passed_items"] == 1
    assert data["summary"]["failed_items"] == 0
    assert data["summary"]["overall_status"] == "PASS"

    item = data["table_4_1"][0]
    expected_fields = [
        "shape",
        "condition",
        "observable",
        "unit",
        "method",
        "quad_cond",
        "scale",
        "scale_str",
        "analytic",
        "analytic_str",
        "reference",
        "reference_str",
        "abs_diff",
        "abs_diff_str",
        "rel_err",
        "rel_err_str",
        "threshold",
        "threshold_str",
        "status",
    ]
    for field in expected_fields:
        assert field in item, f"Field '{field}' missing from JSON table_4_1 item"


def test_example_15_switch_mode_description() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    cmd = [sys.executable, str(repo_root / "examples" / "15_analytic_stage1_benchmarks.py"), "--help"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    help_text = res.stdout
    assert "--report" in help_text
    assert "Switch mode" in help_text
    assert "--json" in help_text
    assert "--markdown" in help_text
