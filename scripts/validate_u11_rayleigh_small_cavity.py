"""Run independent A/B comparisons and separately reread external evidence.

Run under the shared heavy-slot flock, with OMP/OPENBLAS/MKL_NUM_THREADS=1.
The output directory must not exist. No result files belong in Git.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gragra  # noqa: E402
import gragra.analytic.u11_rayleigh_small_cavity as production  # noqa: E402
from tests.oracles.u11_rayleigh_small_cavity_reference import (  # noqa: E402
    elastic_root,
    evaluate,
)

ORDERS = ((40, 12, 1e-12), (64, 20, 1e-13), (96, 32, 2e-14))
PARTS = ("background_halfspace", "cavity_wall_sca", "total_leading_sca")
REFERENCE_PARTS = (
    "rock_bulk",
    "ground_surface",
    *PARTS,
    "wall_full",
    "removed_bulk",
    "finite",
    "wall_variation",
    "model_error",
)
SOURCES = {
    "harms2019.pdf": (
        "Harms 2019 final, LRR 22:6; pp29,43-45",
        "c5a20ecaaffc6f2092c1c310b5e0a3d84258f0b882c0c89133a20c5f96c6872d",
    ),
    "harms2022.pdf": (
        "Harms et al. 2022 EPJ Plus 137:687; p4 Eq7",
        "8a94d532630f5b634bb3b139dda41a566278b566c4643a64c5917bb263aea111",
    ),
    "tempel-2025-slides.pdf": (
        "Tempel/Stahl/Pretz 2025-11-12; slides8,9,42,43",
        "e76e346f0be60adc7384905e1bd303545ba5bf3f7f4b3fe093647da45ee88a7c",
    ),
    "reumers-arxiv-2603.15157v4.pdf": (
        "Reumers et al. arXiv2603.15157v4 2026-04-15",
        "366e340873445b6d3b6e563bac6c69eab82c88122e63f660f239646974fd4921",
    ),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def gate(error, reference, scale, factor=1.0):
    """Per-target vector norm AND component conditions; nonfinite fails."""
    norm = np.linalg.norm(error, axis=-1)
    norm_limit = factor * (1e-12 * scale + 1e-10 * np.linalg.norm(reference, axis=-1))
    component = np.abs(error)
    component_limit = factor * (1e-12 * scale + 1e-10 * np.abs(reference))
    passed = (
        np.isfinite(norm)
        & (norm <= norm_limit)
        & np.all(np.isfinite(component) & (component <= component_limit), axis=-1)
    )
    return {
        "pass": passed.tolist(),
        "norm": norm.tolist(),
        "norm_limit": norm_limit.tolist(),
        "component": component.tolist(),
        "component_limit": component_limit.tolist(),
        "failed_target_component": np.argwhere(
            ~np.isfinite(component) | (component > component_limit)
        ).tolist(),
    }


def fixtures():
    k = 2 * np.pi * 5 / elastic_root(5800.0, 3200.0)[0]
    a = 0.37
    b = 0.29
    ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    rz = np.array([[np.cos(b), -np.sin(b), 0], [np.sin(b), np.cos(b), 0], [0, 0, 1]])
    rot = rz @ ry
    for shape in ("sphere", "cuboid", "cylinder"):
        for kh in (0.2, 1.0, 3.0):
            for eps in (0.1, 0.05, 0.025):
                length = eps / k
                args = {
                    "cavity_shape": shape,
                    "center_m": [13.0, -7.0, -kh / k],
                    "density_kg_m3": 2300.0,
                    "v_p_m_s": 5800.0,
                    "v_s_m_s": 3200.0,
                    "frequency_hz": 5.0,
                    "vertical_surface_amplitude_m": (1 + 0.3j) * 1e-9,
                    "direction": [1.0, 2.0, 0.0],
                    "phase_origin_m": [-4.0, 9.0, 0.0],
                    "rotation": rot,
                }
                if shape == "sphere":
                    args["radius_m"] = length
                    offset = length * np.array([0.23, -0.17, 0.31])
                elif shape == "cuboid":
                    half = (
                        length
                        * np.array([2.0, 1.5, 1.0])
                        / np.linalg.norm([2.0, 1.5, 1.0])
                    )
                    args["half_lengths_m"] = half
                    offset = half * np.array([0.23, -0.17, 0.31])
                else:
                    args["radius_m"] = length / np.sqrt(1 + 0.75**2)
                    args["height_m"] = 1.5 * args["radius_m"]
                    offset = np.array(
                        [
                            0.23 * args["radius_m"],
                            -0.17 * args["radius_m"],
                            0.31 * args["height_m"] / 2,
                        ]
                    )
                targets = np.array(
                    [args["center_m"], np.asarray(args["center_m"]) + rot @ offset]
                )
                yield shape, kh, eps, targets, args


def jsonable(value):
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


def stress_reference(raw):
    """Preserve a hard cuboid query without weakening the ordinary fixture gate."""
    _, _, _, targets, args = next(f for f in fixtures() if f[0] == "cuboid")
    half = np.asarray(args["half_lengths_m"])
    targets[1] = np.asarray(args["center_m"]) + args["rotation"] @ (
        half * np.array([0.85, -0.31, 0.47])
    )
    history = []
    previous = None
    for order, radial, tolerance in ORDERS:
        ref = evaluate(targets, args, order, radial, tolerance)
        for key, value in ref.items():
            raw[f"stress_o{order}_{key}"] = value
        if previous is not None:
            history.append(
                {
                    "order": order,
                    "parts": {
                        key: gate(
                            ref[key] - previous[key], ref[key], 4 * np.pi * 2300e-9, 0.1
                        )
                        for key in REFERENCE_PARTS
                    },
                }
            )
        previous = ref
    converged = all(all(v["pass"]) for v in history[-1]["parts"].values())
    return jsonable(
        {
            "status": "PASS" if converged else "REFERENCE_NOT_CONVERGED",
            "interpretation": "reference stress query, not an implementation verdict",
            "targets_m": targets,
            "inputs": args,
            "refinement": history,
        }
    )


def readback(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    raw = directory / "raw.npz"
    assert digest(raw.read_bytes()) == manifest["raw_sha256"]
    with np.load(raw, allow_pickle=False) as data:
        assert set(data.files) == set(manifest["arrays"])
        for key, record in manifest["arrays"].items():
            value = data[key]
            assert list(value.shape) == record["shape"]
            assert str(value.dtype) == record["dtype"]
            assert np.isfinite(value).all()
            assert digest(value.tobytes()) == record["value_sha256"]
        for index in range(manifest["sample_count"]):
            prefix = f"s{index}_o96_"
            assert np.max(np.abs(data[prefix + "closure"])) < 1e-18
    print(
        f"PASS separate-process readback: {len(manifest['arrays'])} arrays; "
        "hash/shape/dtype/finite/value/closure"
    )


def run(directory, source_dir):
    assert Path(gragra.__file__).resolve().is_relative_to(ROOT / "src")
    assert (
        Path(production.__file__).resolve()
        == ROOT / "src/gragra/analytic/u11_rayleigh_small_cavity.py"
    )
    assert sys.prefix != sys.base_prefix, "use an isolated venv"
    assert Path(sys.executable).parent == Path(sys.prefix) / "bin"
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        assert os.environ.get(var) == "1", f"set {var}=1"
    directory.mkdir(parents=True, exist_ok=False)
    sources = {}
    for name, (version, expected) in SOURCES.items():
        path = source_dir / name
        actual = digest(path.read_bytes())
        assert actual == expected
        sources[name] = {"path": str(path), "version": version, "sha256": actual}
    raw = {}
    samples = []
    acceptance = []
    convergence = []
    model_norm = []
    scale = 4 * np.pi * 2300 * 1e-9
    for index, (shape, kh, eps, targets, args) in enumerate(fixtures()):
        parts, diag = production.rayleigh_small_cavity_acceleration_g_free(
            targets, **args, return_diagnostics=True
        )
        samples.append(
            jsonable(
                {
                    "index": index,
                    "shape": shape,
                    "kh": kh,
                    "eps": eps,
                    "targets_m": targets,
                    "inputs": args,
                    "diagnostics": diag,
                }
            )
        )
        for key, value in parts.items():
            raw[f"s{index}_production_{key}"] = value
        previous = None
        for order, radial, tolerance in ORDERS:
            ref = evaluate(targets, args, order, radial, tolerance)
            for key, value in ref.items():
                raw[f"s{index}_o{order}_{key}"] = value
            if previous is not None:
                convergence.append(
                    {
                        "sample": index,
                        "order": order,
                        "parts": {
                            key: gate(ref[key] - previous[key], ref[key], scale, 0.1)
                            for key in REFERENCE_PARTS
                        },
                    }
                )
            previous = ref
        acceptance.append(
            {
                "sample": index,
                "parts": {
                    key: gate(parts[key] - ref[key], ref[key], scale) for key in PARTS
                },
            }
        )
        for key in PARTS:
            raw[f"s{index}_implementation_error_{key}"] = parts[key] - ref[key]
        model_norm.append(np.linalg.norm(ref["model_error"], axis=-1))
    final_convergence = [c for c in convergence if c["order"] == 96]
    a_ref_ok = all(
        all(c["parts"][key]["pass"]) for c in final_convergence for key in PARTS
    )
    b_ref_ok = all(
        all(c["parts"][key]["pass"])
        for c in final_convergence
        for key in REFERENCE_PARTS
    )
    implementation_ok = all(
        all(c["parts"][key]["pass"]) for c in acceptance for key in PARTS
    )
    norms = np.array(model_norm)
    observed = []
    for start in range(0, len(samples), 3):
        with np.errstate(divide="ignore", invalid="ignore"):
            order = np.log(
                norms[start : start + 2] / norms[start + 1 : start + 3]
            ) / np.log(2)
        observed.append(
            {
                "samples": list(range(start, start + 3)),
                "norm": norms[start : start + 3].tolist(),
                "normalized": (norms[start : start + 3] / scale).tolist(),
                "observed_order": order.tolist(),
                "interpretation": "C: prescribed-field model difference"
                if a_ref_ok and b_ref_ok
                else "REFERENCE_NOT_CONVERGED: do not infer model order",
            }
        )
    stress = stress_reference(raw)
    np.savez_compressed(directory / "raw.npz", **raw)
    manifest = {
        "code_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ),
        "owned_file_sha256": {
            str(path.relative_to(ROOT)): digest(path.read_bytes())
            for path in (
                ROOT / "src/gragra/analytic/u11_rayleigh_small_cavity.py",
                ROOT / "src/gragra/observables/u11_rayleigh_small_cavity.py",
                ROOT / "tests/test_u11_rayleigh_small_cavity.py",
                ROOT / "tests/oracles/u11_rayleigh_small_cavity_reference.py",
                ROOT / "scripts/check_u11_rayleigh_small_cavity_symbolic.py",
                ROOT / "scripts/validate_u11_rayleigh_small_cavity.py",
                ROOT / "docs/design/u11-rayleigh-small-cavity.md",
                ROOT / "docs/validation/u11-rayleigh-small-cavity.md",
            )
            if path.exists()
        },
        "command": [sys.executable, *sys.argv],
        "cwd": str(ROOT),
        "python": platform.python_version(),
        "sys_executable": sys.executable,
        "sys_path": sys.path,
        "gragra_file": gragra.__file__,
        "module_file": production.__file__,
        "dependencies": {
            p: importlib.metadata.version(p)
            for p in ("numpy", "scipy", "mpmath", "sympy", "pytest", "ruff", "mkdocs")
        },
        "cpu_count": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "memory": Path("/proc/meminfo").read_text(),
        "limits": {
            k: resource.getrlimit(getattr(resource, k))
            for k in ("RLIMIT_AS", "RLIMIT_CPU", "RLIMIT_STACK", "RLIMIT_NOFILE")
        },
        "threads": {
            v: os.environ[v]
            for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "sources": sources,
        "settings": ORDERS,
        "units": {
            "acceleration": "kg/m^2",
            "displacement": "m",
            "density": "kg/m^3",
            "frequency": "Hz",
            "coordinates": "m",
            "fourier_quad_error": "dimensionless",
        },
        "scale": scale,
        "sample_count": len(samples),
        "samples": samples,
        "acceptance": acceptance,
        "refinement": convergence,
        "comparison_A": "PASS"
        if a_ref_ok and implementation_ok
        else ("REFERENCE_NOT_CONVERGED" if not a_ref_ok else "IMPLEMENTATION_MISMATCH"),
        "comparison_B": "QUANTIFIED"
        if a_ref_ok and b_ref_ok
        else "REFERENCE_NOT_CONVERGED",
        "model_series": observed,
        "cuboid_stress_reference": stress,
        "review_status": "REVIEW_NOT_RETURNED",
        "arrays": {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "value_sha256": digest(value.tobytes()),
            }
            for key, value in raw.items()
        },
        "raw_sha256": digest((directory / "raw.npz").read_bytes()),
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False)
    )
    with (directory / "readback.log").open("x") as log:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--readback",
                str(directory),
            ],
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in ("code_sha", "sample_count", "comparison_A", "comparison_B")
            },
            indent=2,
        )
    )
    print(f"raw arrays={len(raw)}; separate-process readback PASS")
    return 0 if a_ref_ok and b_ref_ok and implementation_ok else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--readback", type=Path)
    args = parser.parse_args()
    if args.readback:
        readback(args.readback)
        return 0
    if args.output_dir is None or args.source_dir is None:
        parser.error("--output-dir and --source-dir required")
    return run(args.output_dir, args.source_dir)


if __name__ == "__main__":
    raise SystemExit(main())
