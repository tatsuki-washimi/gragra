"""Generate the P6 scientific-validation evidence (JSON and Markdown).

Runs the V1-V8 measurements through the helpers in
``tests/test_m3_p6_science.py`` so the report and the pytest assertions use one
code path. The thresholds are the module constants fixed before the first run.

Outputs ``p6-report.json`` and ``p6-report.md`` in the directory given as the
first argument (default ``./p6-report``). Keep generated evidence out of git.

Run from the repository root with the project environment:
    python scripts/m3_p6_report.py <output-directory>
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

WT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WT / "src"))
sys.path.insert(0, str(WT / "tests"))

import gragra  # noqa: E402

_GRAGRA_FILE = Path(gragra.__file__).resolve()
if not _GRAGRA_FILE.is_relative_to((WT / "src").resolve()):
    raise SystemExit(f"gragra imported from outside this worktree: {_GRAGRA_FILE}")

import astropy  # noqa: E402
import mpmath  # noqa: E402
import numpy as np  # noqa: E402
import scipy  # noqa: E402
import test_m3_p6_science as P  # noqa: E402, N812
from astropy import units as u  # noqa: E402
from gwpy.time import LIGOTimeGPS  # noqa: E402

import gragra.observables.rigid_motion as rigid_motion_module  # noqa: E402

OUT_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "p6-report")


def _verdict(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _guard(fn):
    """Run a measurement; record exceptions as data instead of aborting the report."""
    try:
        return {"status": "ok", "data": fn()}
    except Exception as exc:  # recorded for the report
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def section_v1() -> dict:
    rows = P.measure_v1()
    for row in rows:
        row["verdict"] = _verdict(row["max_normalised_error"] <= P.THR_V1)
    return {
        "threshold": P.THR_V1,
        "rows": rows,
        "verdict": _verdict(all(r["verdict"] == "PASS" for r in rows)),
    }


def section_v2() -> dict:
    per_obs = []
    for name in ("potential", "acceleration", "gradient"):
        row = _guard(lambda name=name: P.measure_v2(name))
        if row["status"] == "ok":
            row["verdict"] = _verdict(row["data"]["max_normalised_error"] <= P.THR_V2)
        else:
            row["verdict"] = "FAIL"
        row["observable"] = name
        per_obs.append(row)
    closed = _guard(P.measure_v2_closed_form)
    quad = _guard(P.measure_v2_quadrature_independence)
    conv = _guard(P.measure_v2_convergence)
    return {
        "threshold": P.THR_V2,
        "oracle_mpmath_dps": 50,
        "per_observable": per_obs,
        "closed_form_centre_target": closed,
        "oracle_quadrature_independence": quad,
        "convergence": conv,
        "verdict": _verdict(
            all(r.get("verdict") == "PASS" for r in per_obs)
            and closed["status"] == "ok"
            and closed["data"]["max_normalised_error_n0_3"] <= P.THR_V2
        ),
    }


def section_v3() -> dict:
    rows = [_guard(lambda b=b: P.measure_v3(b)) for b in P.V3_BS]
    edge = _guard(P.measure_v3_edge)
    conv = _guard(P.measure_v3_convergence)
    ok_rows = [
        r["data"]["max_normalised_error"] <= P.THR_V3
        for r in rows
        if r["status"] == "ok"
    ]
    return {
        "threshold": P.THR_V3,
        "passages": rows,
        "edge_case": edge,
        "convergence": conv,
        "verdict": _verdict(len(ok_rows) == len(rows) and all(ok_rows)),
    }


def section_v4() -> dict:
    rows = [_guard(lambda n=n: P.measure_v4(n)) for n in (512, 1024, 2048)]
    ok = [
        r["data"]["max_normalised_error_n_le_8"] <= P.THR_V4
        for r in rows
        if r["status"] == "ok"
    ]
    return {
        "threshold": P.THR_V4,
        "rows": rows,
        "verdict": _verdict(len(ok) == 3 and all(ok)),
    }


def section_v5() -> dict:
    units = _guard(P.measure_v5_units)
    frame = _guard(P.measure_v5_frame)
    phase = _guard(P.measure_v5_phase)
    epoch = _guard(P.measure_v5_epoch)
    unit_ok = units["status"] == "ok" and all(
        "error" not in row and P._same_unit(row["unit"], row["expected"])
        for row in units["data"].values()
    )
    numeric_ok = all(
        r["status"] == "ok" and r["data"]["max_normalised_error"] <= P.THR_V5
        for r in (frame, phase, epoch)
    )
    return {
        "threshold": P.THR_V5,
        "units": units,
        "frame_basis": frame,
        "phase_convention": phase,
        "epoch_shift": epoch,
        "verdict": _verdict(unit_ok and numeric_ok),
    }


def section_v6() -> dict:
    base = _guard(lambda: P.measure_v6())
    original = rigid_motion_module.G_SI
    try:
        rigid_motion_module.G_SI = 2.0 * original
        doubled = _guard(lambda: P.measure_v6())
    finally:
        rigid_motion_module.G_SI = original
    ok_once = base["status"] == "ok"
    if ok_once:
        got = np.asarray(base["data"]["got"])
        ref = np.asarray(base["data"]["ref"])
        s = P.scale_si(2.0, 1.0, 2)
        err_once = float(np.max(np.abs(got - ref)) / s)
        base["data"]["normalised_error_vs_point_acceleration"] = err_once
    ratio = None
    if ok_once and doubled["status"] == "ok":
        got2 = np.asarray(doubled["data"]["got"])
        ratio = float(
            np.max(np.abs(got2 - 2.0 * np.asarray(base["data"]["got"])))
            / P.scale_si(2.0, 1.0, 2)
        )
    verdict = (
        ok_once and base["data"]["normalised_error_vs_point_acceleration"] <= P.THR_V6
    )
    verdict = verdict and ratio is not None and ratio <= P.THR_V6
    return {
        "threshold": P.THR_V6,
        "g_applied_once": base,
        "g_doubled": doubled,
        "normalised_deviation_from_2x": ratio,
        "verdict": _verdict(verdict),
    }


def section_v7() -> dict:
    tiny = None
    try:
        P._periodic_run(1, 3, budget=1024)
        tiny = {"raised": False}
    except MemoryError as exc:
        tiny = {"raised": True, "message": str(exc)}
    except Exception as exc:
        tiny = {"raised": False, "unexpected": f"{type(exc).__name__}: {exc}"}
    default_run = _guard(lambda: P._values(P._periodic_run(1, 6), "acceleration").shape)
    grid = _guard(P.measure_v7_memory_grid)
    timing = _guard(P.measure_v7_timing)
    verdict = tiny.get("raised") is True and default_run["status"] == "ok"
    return {
        "budget_scope": "managed arrays, not process RSS",
        "tiny_budget_1024_bytes": tiny,
        "default_budget": default_run,
        "tracemalloc_grid": grid,
        "timing": timing,
        "verdict": _verdict(verdict),
        "note": "Timing and grid are recorded only; the pass criterion is MemoryError.",
    }


def _v8_record(name, fn, expect):
    """Run an applicability probe and record what happened (exception or result)."""
    try:
        fn()
        outcome = {"outcome": "returned"}
    except Exception as exc:
        outcome = {
            "outcome": "exception",
            "type": type(exc).__name__,
            "message": str(exc)[:300],
        }
        if hasattr(exc, "reason"):
            outcome["reason"] = exc.reason
    outcome["name"] = name
    outcome["expected"] = expect
    reason_ok = "reason" not in expect or outcome.get("reason") == expect["reason"]
    outcome["match"] = bool(
        outcome["outcome"] == "exception"
        and outcome.get("type") in expect["types"]
        and reason_ok
    )
    return outcome


def section_v8() -> dict:
    model_circle = P.ContinuousFourierModel(
        P._circle_source(), P.circular_pose(), frame=P.WORLD
    )
    from _continuous_fixtures import constant_velocity_pose

    nonperiodic = P.ContinuousFourierModel(
        P._circle_source(),
        constant_velocity_pose(velocity_m_s=(1.0, 0.0, 0.0), origin_m=(0.0, 1.0, 0.0)),
        frame=P.WORLD,
    )
    records = [
        _v8_record(
            "P4 non-periodic motion",
            lambda: P._periodic(
                nonperiodic,
                P._framed([[0.0, 0.0, 0.0]]),
                orders=np.arange(0, 3),
                observables=("acceleration",),
            ).run(),
            {"types": ["ContinuousQuadratureError"], "reason": "not_periodic"},
        ),
        _v8_record(
            "P4 tau_min missing",
            lambda: P._periodic(
                model_circle,
                P._framed([[0.0, 0.0, 0.0]]),
                orders=np.arange(0, 3),
                observables=("acceleration",),
                tau_min=None,
            ).run(),
            {"types": ["ValueError"]},
        ),
        _v8_record(
            "P5 f_max*|Delta| exceeded",
            lambda: model_circle.window_at(
                P._framed([[0.0, 0.0, 0.0]]),
                epoch=LIGOTimeGPS(1_000_000_000, 0),
                tref=LIGOTimeGPS(0, 0),
                t_start=-1.0 * u.s,
                t_stop=1.0 * u.s,
                frequencies=np.array([2e-3]) * u.Hz,
                tau_min=0.25 * u.s,
                atol=1e-10 * u.m / u.s,
                rtol=1e-10,
                max_evaluations=P.DEFAULT_MAX_EVALUATIONS,
                memory_budget_bytes=P.MEM,
            ).run(),
            {"types": ["ValueError"]},
        ),
        _v8_record(
            "P5 max_evaluations insufficient",
            lambda: (
                P._passage_model(0.5)
                .window_at(
                    P._framed([[0.0, 0.0, 0.0]]),
                    epoch=P.EPOCH0,
                    tref=P.EPOCH0,
                    t_start=-3.0 * u.s,
                    t_stop=3.0 * u.s,
                    frequencies=np.asarray(P.V3_FREQS) * u.Hz,
                    tau_min=0.25 * u.s,
                    atol=1e-10 * u.m / u.s,
                    rtol=1e-10,
                    max_evaluations=10,
                    memory_budget_bytes=P.MEM,
                )
                .run()
            ),
            {"types": ["ValueError", "RuntimeError"]},
        ),
        _v8_record(
            "non-finite pose",
            lambda: P._periodic(
                P.ContinuousFourierModel(
                    P._circle_source(),
                    lambda times: (
                        P.Rotation.identity(len(times)),
                        np.full((len(times), 3), np.nan) * u.m,
                    ),
                    frame=P.WORLD,
                ),
                P._framed([[0.0, 0.0, 0.0]]),
                orders=np.arange(0, 3),
                observables=("acceleration",),
            ).run(),
            {
                "types": [
                    "ValueError",
                    "RuntimeError",
                    "TypeError",
                    "FloatingPointError",
                ]
            },
        ),
    ]
    passage = _guard(P.measure_v8_passage_tau_profile)
    characterization = _guard(P.measure_characterization_gaussian)
    passage_ok = passage["status"] == "ok" and all(
        r["outcome"] == "returned" and r["max_normalised_error"] <= P.THR_V3
        for r in passage["data"]
    )
    return {
        "negative_probes": records,
        "offgrid_close_passage_tau_profile": passage,
        "characterization_gaussian": characterization,
        "verdict": _verdict(all(r["match"] for r in records) and passage_ok),
    }


def _provenance() -> dict:
    """Record the code state the numbers came from (HEAD and dirty status)."""
    import subprocess

    def git(*args: str) -> str:
        proc = subprocess.run(
            ["git", "-C", str(WT), *args],
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.stdout.strip() if proc.returncode == 0 else f"error: {proc.stderr}"

    porcelain = git("status", "--porcelain")
    return {
        "git_head": git("rev-parse", "HEAD"),
        "dirty": bool(porcelain) and not porcelain.startswith("error"),
        "status_porcelain": porcelain.splitlines(),
    }


def _item(item: str, fn, extract) -> list[dict]:
    """Run one measurement and turn it into report rows.

    ``extract(data)`` yields ``(metric, value, threshold, ok)`` tuples. A failed
    measurement is reported as a FAIL row, not hidden.
    """
    res = _guard(fn)
    if res["status"] != "ok":
        return [
            {
                "item": item,
                "metric": "measurement",
                "value": None,
                "threshold": None,
                "verdict": "FAIL",
                "error": res["error"],
            }
        ]
    rows = []
    for metric, value, thr, ok in extract(res["data"]):
        rows.append(
            {
                "item": item,
                "metric": metric,
                "value": value,
                "threshold": thr,
                "verdict": _verdict(bool(ok)) if ok is not None else "RECORDED",
            }
        )
    return rows


def section_m() -> dict:
    """M1-M3 additions: independent oracles, convergence, negative controls."""
    rows: list[dict] = []
    rows += _item(
        "M1 a",
        P.measure_v1_independent,
        lambda d: [
            (
                f"{r['observable']} max|err|/S (unsorted, duplicate times)",
                r["max_normalised_error"],
                P.THR_V1_INDEP,
                r["max_normalised_error"] <= P.THR_V1_INDEP,
            )
            for r in d["rows"]
        ],
    )
    for obs in ("potential", "acceleration", "gradient"):
        rows += _item(
            "M1 b",
            lambda o=obs: P.measure_v2_rotating(o),
            lambda d: [
                (
                    f"{d['observable']} max|err|/S (rotating body vs mpmath)",
                    d["max_normalised_error"],
                    P.THR_V2_ROT,
                    d["max_normalised_error"] <= P.THR_V2_ROT,
                )
            ],
        )
    rows += _item(
        "M1 c",
        P.measure_v2_gradient_structure,
        lambda d: [
            (
                "gradient symmetry max|T_ij - T_ji|/S",
                d["symmetry_normalised"],
                P.THR_IDENT,
                d["symmetry_normalised"] <= P.THR_IDENT,
            ),
            (
                "gradient trace max|tr T|/S",
                d["trace_normalised"],
                P.THR_IDENT,
                d["trace_normalised"] <= P.THR_IDENT,
            ),
        ],
    )

    def fft_extract(rs):
        e = [r["max_normalised_error_n_le_8"] for r in rs]
        return [
            ("FFT vs direct, N=128 (reference)", e[0], None, None),
            ("FFT vs direct, N=256", e[1], e[0] / 10, e[1] <= e[0] / 10),
            (
                "FFT vs direct, N=512",
                e[2],
                min(e[1] / 10, 1e-9),
                e[2] <= e[1] / 10 and e[2] <= 1e-9,
            ),
        ]

    rows += _item(
        "M2 e",
        lambda: [P.measure_v4(n, target=(1.1, 0.0, 0.0)) for n in (128, 256, 512)],
        fft_extract,
    )
    rows += _item(
        "M2 h",
        P.measure_v8_passage_tau_profile,
        lambda rs: [
            (
                f"off-grid close passage, tau_min={r['tau_min_s']} s",
                r.get("max_normalised_error"),
                P.THR_V3,
                r["outcome"] == "returned"
                and r.get("max_normalised_error", np.inf) <= P.THR_V3,
            )
            for r in rs
        ],
    )

    def char_extract(rs):
        fine, coarse = rs
        return [
            (
                "Gaussian tau_min=1e-3 s relative error (must be <= THR_V3)",
                fine["relative_error"],
                P.THR_V3,
                fine["reason"] is None and fine["relative_error"] <= P.THR_V3,
            ),
            (
                "Gaussian tau_min=1.0 s relative error (documented limit, >0.5)",
                coarse["relative_error"],
                0.5,
                coarse["reason"] is None and coarse["relative_error"] > 0.5,
            ),
        ]

    rows += _item(
        "M2 h(2) characterization",
        P.measure_characterization_gaussian,
        char_extract,
    )
    rows += _item(
        "M2 k",
        P.measure_v5_closed_form_nonzero_delta,
        lambda d: [
            (
                "Delta=0.1 s complex n=0..3 max|err|/S",
                d["max_normalised_error_complex_n0_3"],
                P.THR_V5,
                d["max_normalised_error_complex_n0_3"] <= P.THR_V5,
            ),
            (
                "Delta=0.1 s |A_1| vs closed form",
                d["modulus_A1_minus_closed_form_normalised"],
                P.THR_V5,
                d["modulus_A1_minus_closed_form_normalised"] <= P.THR_V5,
            ),
        ],
    )
    rows += _item(
        "M1 j",
        P.measure_v5_frame_chain,
        lambda d: [
            (
                f"frame chain {k} max|err|/S",
                v,
                P.THR_FRAME,
                v <= P.THR_FRAME,
            )
            for k, v in d.items()
            if k != "threshold"
        ],
    )

    def near_extract(d):
        out = []
        for r in d["rows"]:
            out.append(
                (
                    f"d={d['distance_m']} m rtol={r['rtol']:g} (nodes={r['nodes']})",
                    r["max_normalised_error"],
                    None,
                    None,
                )
            )
        errs = [r["max_normalised_error"] for r in d["rows"]]
        nodes = [r["nodes"] for r in d["rows"]]
        mono = all(
            b <= a + P.THR_V2_CONV_SLACK for a, b in zip(errs, errs[1:], strict=False)
        )
        out.append(
            (
                f"d={d['distance_m']} m monotone, final<=THR_V2, nodes non-decreasing",
                errs[-1],
                P.THR_V2,
                mono and errs[-1] <= P.THR_V2 and nodes[0] <= nodes[-1],
            )
        )
        return out

    for dist in P.NEAR_DISTANCES_M:
        rows += _item(
            "M3 d",
            lambda d=dist: P.measure_d_near_rtol(d),
            near_extract,
        )
    rows += _item(
        "M3 f",
        P.measure_v3_convergence,
        lambda d: [
            (
                f"tau_min partition {k} (S*W)",
                v,
                P.THR_V3,
                v <= P.THR_V3,
            )
            for k, v in d["tau_min_partition_normalised"].items()
        ],
    )
    rows += _item(
        "M3 g",
        P.measure_v3_negative_controls,
        lambda d: (
            [
                (
                    "V3 correct answer accepted",
                    d["correct"],
                    P.THR_V3,
                    d["correct"] <= P.THR_V3,
                )
            ]
            + [
                (f"V3 wrong ({k}) rejected", v, P.THR_NEG, v > P.THR_NEG)
                for k, v in d["wrong"].items()
            ]
        ),
    )
    rows += _item(
        "M3 g",
        P.measure_v2_negative_controls,
        lambda d: (
            [
                (
                    "V2 correct answer accepted",
                    d["correct"],
                    P.THR_V2,
                    d["correct"] <= P.THR_V2,
                )
            ]
            + [
                (f"V2 wrong ({k}) rejected", v, P.THR_NEG, v > P.THR_NEG)
                for k, v in d["wrong"].items()
            ]
        ),
    )

    def v6_neg():
        base = P.measure_v6()
        s = P.scale_si(2.0, 1.0, 2)
        got = np.asarray(base["got"], dtype=np.float64)
        ref = np.asarray(base["ref"], dtype=np.float64)
        return {
            "correct": P.normalised_error(got, ref, s),
            "double_G": P.normalised_error(
                got, ref * float(P.gravitational_constant()), s
            ),
        }

    rows += _item(
        "M3 g",
        v6_neg,
        lambda d: [
            (
                "V6 correct answer accepted",
                d["correct"],
                P.THR_V6,
                d["correct"] <= P.THR_V6,
            ),
            (
                "V6 double-G rejected",
                d["double_G"],
                P.THR_NEG,
                d["double_G"] > P.THR_NEG,
            ),
        ],
    )
    verdicts = [r["verdict"] for r in rows if r["verdict"] in ("PASS", "FAIL")]
    return {
        "rows": rows,
        "verdict": _verdict(bool(verdicts) and all(v == "PASS" for v in verdicts)),
    }


def _environment() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "astropy": astropy.__version__,
        "mpmath": mpmath.__version__,
        "gragra_file": str(_GRAGRA_FILE),
        "gragra_src_is_worktree": True,
        "worktree": str(WT),
    }


def _md_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _fmt(x) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.3e}"
    return str(x)


def render_markdown(report: dict) -> str:
    out = [
        "# P6 scientific validation report",
        "",
        f"Generated: {report['generated_utc']}",
        "",
        "Scale: S = G M / L^p (p = 1, 2, 3 for potential, acceleration, gradient; "
        "M = 1 kg, L = 1 m). Window: S * W (W in seconds). The budget covers managed "
        "arrays only, not process RSS.",
        "",
        "## Verdict summary",
        "",
    ]
    summary = [(k, v["verdict"]) for k, v in report["sections"].items()]
    out.append(_md_table(["item", "verdict"], summary))
    s = report["sections"]
    out += [
        "",
        "## V1 M3-A agreement",
        "",
        f"Threshold {P.THR_V1:g}. Verdict {s['V1']['verdict']}.",
        "",
        _md_table(
            ["observable", "shape", "max normalised error", "verdict"],
            [
                (
                    r["observable"],
                    r["shape"],
                    _fmt(r["max_normalised_error"]),
                    r["verdict"],
                )
                for r in s["V1"]["rows"]
            ],
        ),
    ]
    v2 = s["V2"]
    out += [
        "",
        "## V2 periodic Fourier",
        "",
        f"Threshold {P.THR_V2:g}. Verdict {v2['verdict']}.",
        "",
        _md_table(
            ["observable", "status", "max normalised error", "verdict", "note"],
            [
                (
                    r["observable"],
                    r["status"],
                    _fmt(r["data"]["max_normalised_error"])
                    if r["status"] == "ok"
                    else "-",
                    r["verdict"],
                    r.get("error", ""),
                )
                for r in v2["per_observable"]
            ],
        ),
    ]
    out += [
        "",
        "Convergence (rtol ladder, acceleration):",
        "",
        _md_table(
            ["rtol", "max normalised error"],
            [
                (_fmt(r["rtol"]), _fmt(r["max_normalised_error"]))
                for r in v2["convergence"]["data"]["rtol_ladder"]
            ],
        )
        if v2["convergence"]["status"] == "ok"
        else str(v2["convergence"]),
    ]
    v3 = s["V3"]
    out += [
        "",
        "## V3 finite window",
        "",
        f"Threshold {P.THR_V3:g}. Verdict {v3['verdict']}.",
        "",
        _md_table(
            ["b (m)", "max normalised error (S*W)"],
            [
                (r["data"]["b_m"], _fmt(r["data"]["max_normalised_error"]))
                for r in v3["passages"]
                if r["status"] == "ok"
            ],
        ),
    ]
    v4 = s["V4"]
    out += [
        "",
        "## V4 direct vs FFT (low orders n <= 8)",
        "",
        f"Threshold {P.THR_V4:g}. Verdict {v4['verdict']}.",
        "",
        _md_table(
            ["N samples", "max normalised error (n <= 8)"],
            [
                (r["data"]["n_samples"], _fmt(r["data"]["max_normalised_error_n_le_8"]))
                for r in v4["rows"]
                if r["status"] == "ok"
            ],
        ),
    ]
    v5 = s["V5"]
    out += [
        "",
        "## V5 units, frame, phase, epoch",
        "",
        f"Threshold {P.THR_V5:g}. Verdict {v5['verdict']}.",
        "",
        "```json",
        json.dumps(
            {
                k: v5[k]
                for k in ("units", "frame_basis", "phase_convention", "epoch_shift")
            },
            indent=2,
            default=str,
        )[:4000],
        "```",
    ]
    v6 = s["V6"]
    out += [
        "",
        "## V6 G applied once",
        "",
        f"Threshold {P.THR_V6:g}. Verdict {v6['verdict']}. "
        f"Deviation from 2x after doubling G_SI: "
        f"{_fmt(v6['normalised_deviation_from_2x'])}.",
        "",
    ]
    v7 = s["V7"]
    out += [
        "## V7 memory and performance",
        "",
        f"Verdict {v7['verdict']} (criterion: MemoryError on a tiny budget, "
        "default budget completes). Budget scope: managed arrays, not RSS.",
        "",
        _md_table(
            ["targets", "orders", "tracemalloc peak (bytes)", "peak / budget"],
            [
                (
                    r["n_targets"],
                    r["n_orders"],
                    r["tracemalloc_peak_bytes"],
                    _fmt(r["peak_over_budget"]),
                )
                for r in v7["tracemalloc_grid"]["data"]
            ],
        )
        if v7["tracemalloc_grid"]["status"] == "ok"
        else str(v7["tracemalloc_grid"]),
        "",
        "Timing (median of 3, recorded only):",
        "",
        "```json",
        json.dumps(v7["timing"], indent=2, default=str),
        "```",
    ]
    v8 = s["V8"]
    out += [
        "",
        "## V8 negative tests and applicability",
        "",
        f"Verdict {v8['verdict']}.",
        "",
        _md_table(
            ["probe", "outcome", "type", "reason", "match"],
            [
                (
                    r["name"],
                    r["outcome"],
                    r.get("type", "-"),
                    r.get("reason", "-"),
                    r["match"],
                )
                for r in v8["negative_probes"]
            ],
        ),
        "",
        "Off-grid close passage (b = 0.01 m, peak at t = 0.37 s), error vs tau_min:",
        "",
        _md_table(
            ["tau_min (s)", "outcome", "max|err|/(S*W)"],
            [
                (
                    r["tau_min_s"],
                    r["outcome"],
                    _fmt(r.get("max_normalised_error")),
                )
                for r in v8["offgrid_close_passage_tau_profile"]["data"]
            ]
            if v8["offgrid_close_passage_tau_profile"]["status"] == "ok"
            else [str(v8["offgrid_close_passage_tau_profile"])],
        ),
        "",
        "Characterization (Gaussian, sigma = 1e-4 s, freqs [0, 1] Hz; not a pass "
        "criterion except the tau_min = 1e-3 point; contract sections 3 and 4.4):",
        "",
        "```json",
        json.dumps(v8["characterization_gaussian"], indent=2, default=str),
        "```",
    ]
    m = s["M"]
    out += [
        "",
        "## M1-M3 additional checks",
        "",
        f"Verdict {m['verdict']}. RECORDED rows carry no pass/fail criterion.",
        "",
        _md_table(
            ["item", "metric", "value", "threshold", "verdict"],
            [
                (
                    r["item"],
                    # Escape pipes so that GFM does not split the cell.
                    r["metric"].replace("|", "\\|"),
                    _fmt(r["value"]),
                    _fmt(r["threshold"]),
                    r["verdict"]
                    if r["verdict"] != "FAIL" or "error" not in r
                    else f"FAIL ({r['error'][:120]})",
                )
                for r in m["rows"]
            ],
        ),
        "",
        "## Provenance",
        "",
        f"Git HEAD: {report['provenance']['git_head']}",
        f"Dirty worktree: {report['provenance']['dirty']}",
        "",
        "```",
        *report["provenance"]["status_porcelain"],
        "```",
    ]
    return "\n".join(out) + "\n"


def main() -> None:
    t0 = time.time()
    sections = {
        "V1": _guard(section_v1),
        "V2": _guard(section_v2),
        "V3": _guard(section_v3),
        "V4": _guard(section_v4),
        "V5": _guard(section_v5),
        "V6": _guard(section_v6),
        "V7": _guard(section_v7),
        "V8": _guard(section_v8),
        "M": _guard(section_m),
    }
    flat = {}
    for key, val in sections.items():
        if val["status"] == "ok":
            flat[key] = val["data"]
        else:
            flat[key] = {"verdict": "FAIL", "error": val["error"]}
    report = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.time() - t0, 2),
        "environment": _environment(),
        "provenance": _provenance(),
        "thresholds": {
            "V1": P.THR_V1,
            "V1_independent": P.THR_V1_INDEP,
            "V2": P.THR_V2,
            "V2_rotating": P.THR_V2_ROT,
            "V2_convergence_slack": P.THR_V2_CONV_SLACK,
            "V3": P.THR_V3,
            "V4": P.THR_V4,
            "V5": P.THR_V5,
            "frame_chain": P.THR_FRAME,
            "gradient_symmetry_trace": P.THR_IDENT,
            "V6": P.THR_V6,
            "negative_control_wrong_answer_min": P.THR_NEG,
            "memory_budget_bytes": P.MEM,
        },
        "sections": flat,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "p6-report.json").write_text(
        json.dumps(report, indent=2, default=str, ensure_ascii=False) + "\n"
    )
    try:
        markdown = render_markdown(report)
    except Exception as exc:  # the JSON is the primary record; keep it either way
        markdown = (
            f"# P6 report\n\nMarkdown rendering failed: {type(exc).__name__}: {exc}\n"
        )
    (OUT_DIR / "p6-report.md").write_text(markdown)
    print(json.dumps({k: v["verdict"] for k, v in flat.items()}, indent=2))


if __name__ == "__main__":
    main()
