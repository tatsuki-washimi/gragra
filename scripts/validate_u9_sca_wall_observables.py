"""U9 independent wall oracle and external evidence with process readback.

Run from an installed isolated environment. --output-dir is a NEW run directory
under the requested code SHA; an existing directory is never overwritten.
Heavy-slot flock and thread caps are the caller's responsibility.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import gragra
from gragra.analytic import u9_sca_wall_observables as production

ROOT = Path(__file__).resolve().parents[1]
# Only the explicit checkout is added for its independent test reference.
sys.path.insert(0, str(ROOT))
from tests.oracles.u9_sca_wall_observables_reference import (  # noqa: E402
    acceptance,
    fixtures,
    review_points,
    wall_reference,
)

GEOMETRY = {
    "sphere": {"radius_m": 2.0},
    "cuboid": {"half_lengths_m": [2.0, 3.0, 5.0]},
    "cylinder": {"radius_m": 2.0, "height_m": 6.0},
}

OWNED = (
    "src/gragra/analytic/u9_sca_wall_observables.py",
    "src/gragra/analytic/_u9_spatial_derivatives.py",
    "src/gragra/observables/u9_sca_wall_observables.py",
    "tests/test_u9_sca_wall_observables.py",
    "tests/oracles/u9_sca_wall_observables_reference.py",
    "scripts/check_u9_sca_wall_observables_symbolic.py",
    "scripts/validate_u9_sca_wall_observables.py",
    "docs/design/u9-sca-wall-observables.md",
    "docs/validation/u9-sca-wall-observables.md",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def readback(directory):
    """Separate process checks archive bytes, each array's values/shape/dtype."""
    manifest = json.loads((directory / "manifest.json").read_text())
    archive = directory / "raw.npz"
    assert digest(archive) == manifest["raw_sha256"]
    with np.load(archive, allow_pickle=False) as raw:
        assert sorted(raw.files) == sorted(manifest["arrays"])
        for name, expected in manifest["arrays"].items():
            value = raw[name]
            assert list(value.shape) == expected["shape"]
            assert str(value.dtype) == expected["dtype"]
            assert np.isfinite(value).all()
            assert hashlib.sha256(value.tobytes()).hexdigest() == expected["sha256"]
    result = {
        "status": "READBACK_PASSED",
        "pid": os.getpid(),
        "raw_sha256": manifest["raw_sha256"],
        "arrays": len(manifest["arrays"]),
        "meaning": "serialization identity, not scientific validity",
    }
    with (directory / "readback.json").open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


def metrics(value, reference, shape, block):
    tolerance = acceptance(reference, shape, block)
    difference = np.abs(value - reference)
    if block:
        axes = tuple(range(-block, 0))
        norm_difference = np.sqrt(np.sum(difference**2, axis=axes))
        norm_reference = np.sqrt(np.sum(np.abs(reference) ** 2, axis=axes))
    else:
        norm_difference, norm_reference = difference, np.abs(reference)
    norm_tolerance = acceptance(norm_reference, shape, block)
    return difference, tolerance, norm_difference, norm_tolerance


def run(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    assert Path(sys.executable).parent.parent == args.venv.resolve()
    assert Path(gragra.__file__).resolve().is_relative_to(ROOT / "src")
    assert (
        Path(production.__file__).resolve()
        == ROOT / "src/gragra/analytic/u9_sca_wall_observables.py"
    )
    # No other worktree or original checkout may enter the import search path.
    for item in sys.path:
        if not item:
            continue
        resolved = Path(item).resolve()
        if (resolved / "gragra" / "__init__.py").is_file():
            assert resolved == ROOT / "src"
        if "/.paseo/worktrees/" in str(resolved):
            assert resolved.is_relative_to(ROOT)
    sha = git("rev-parse", "HEAD")
    assert output.parent.name == sha, "evidence parent must be the evaluated code SHA"
    samples = np.vstack(
        (np.eye(3) * 1e-9, 1e-9 * np.array([1 + 0.4j, 2 - 0.7j, -1 + 0.2j]))
    )
    settings = [
        (96, 384),
        (128, 384),
        (128, 512),
        (192, 512),
        (192, 768),
        (256, 768),
        (256, 1024),
    ]
    manifest = {
        "code_sha": sha,
        "base_sha": "f665ee6c500001721966d161ea04ffdfff72f827",
        "dirty": git("status", "--porcelain"),
        "owned_files_sha256": {name: digest(ROOT / name) for name in OWNED},
        "python": sys.version,
        "executable": sys.executable,
        "sys_path": sys.path,
        "gragra_file": gragra.__file__,
        "new_module_file": production.__file__,
        "deps": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "sympy", "mpmath", "pytest", "ruff")
        },
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "command": sys.argv,
        "geometry": GEOMETRY,
        "density_kg_m3": 2300,
        "units": ["kg/m", "kg/m^2", "kg/m^3"],
        "settings": settings,
        "production": {
            "axis_ratio": 0.1,
            "axis_n_max": 5,
            "far_enclosing_radius_ratio": 16,
            "far_degree_max": 8,
        },
        "oracle_shared": "inputs/conventions only; reference has no gragra import",
        "error_categories": {
            "numerical_implementation": "component and norm comparisons",
            "quadrature": "independent Gauss and angular refinement",
            "series_truncation": "switch fixtures plus analytic design",
            "thin_shell_cancellation": "NOT_APPLICABLE",
            "model_approximation": "NOT_EVALUATED; uniform wall SCA only",
        },
        "sources": {},
        "diagnostics": [],
        "review": "REVIEW_NOT_RETURNED",
    }
    for source in args.source_dir.glob("*.pdf"):
        manifest["sources"][source.name] = {
            "path": str(source),
            "sha256": digest(source),
        }
    raw = {"displacement_m": samples}
    all_ok = True
    for shape in GEOMETRY:
        points = np.concatenate((fixtures(shape), review_points(shape)))
        raw[shape + "_targets_m"] = points
        function = getattr(production, shape + "_sca_wall_observables_g_free")
        actual = tuple(
            function(
                points,
                center_m=[0, 0, 0],
                density_kg_m3=2300,
                displacement_m=samples,
                **GEOMETRY[shape],
            )
        )
        references = []
        for order, angular in settings:
            ref = wall_reference(
                shape, points, samples, order=order, angular_order=angular
            )
            references.append(ref)
            for block, value in enumerate(ref):
                raw[f"{shape}_ref_{order}_{angular}_{block}"] = value
        final = references[-1]
        for block, (value, reference) in enumerate(zip(actual, final, strict=True)):
            prefix = f"{shape}_block{block}"
            raw[prefix + "_actual"] = value
            error, tolerance, norm_error, norm_tolerance = metrics(
                value, reference, shape, block
            )
            raw[prefix + "_error"] = error
            raw[prefix + "_tolerance"] = tolerance
            # Keep both final one-coordinate refinement differences. Use ALL
            # components and samples; no aggregate-only convergence decision.
            refinements = []
            for before, after in ((-3, -2), (-2, -1)):
                d, _, nd, _ = metrics(
                    references[before][block], references[after][block], shape, block
                )
                refinements.append((d, nd))
                raw[prefix + f"_refinement_{before}_{after}"] = d
            for sample in range(len(samples)):
                for query in range(len(points)):
                    component_ok = bool(
                        np.all(error[sample, query] <= tolerance[sample, query])
                    )
                    norm_ok = bool(
                        norm_error[sample, query] <= norm_tolerance[sample, query]
                    )
                    converged = all(
                        np.all(d[sample, query] <= tolerance[sample, query] / 10)
                        and nd[sample, query] <= norm_tolerance[sample, query] / 10
                        for d, nd in refinements
                    )
                    status = (
                        "PASS"
                        if converged and component_ok and norm_ok
                        else (
                            "REFERENCE_NOT_CONVERGED"
                            if not converged
                            else "IMPLEMENTATION_MISMATCH"
                        )
                    )
                    all_ok &= status == "PASS"
                    manifest["diagnostics"].append(
                        {
                            "shape": shape,
                            "sample": sample,
                            "query": query,
                            "block": block,
                            "status": status,
                            "component_error": error[sample, query].tolist(),
                            "component_tolerance": tolerance[sample, query].tolist(),
                            "norm_error": float(norm_error[sample, query]),
                            "norm_tolerance": float(norm_tolerance[sample, query]),
                            "max_error_ratio": float(
                                np.max(error[sample, query] / tolerance[sample, query])
                            ),
                            "refinement_ratios": [
                                float(
                                    np.max(
                                        d[sample, query]
                                        / (tolerance[sample, query] / 10)
                                    )
                                )
                                for d, _ in refinements
                            ],
                        }
                    )
        print(shape, "evaluated", len(points), "targets", flush=True)
    manifest["status"] = "PASS" if all_ok else "FAILED"
    archive = output / "raw.npz"
    np.savez_compressed(archive, **raw)
    manifest["raw_sha256"] = digest(archive)
    manifest["arrays"] = {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        }
        for name, value in raw.items()
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False)
    )
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--readback", str(output)],
        check=True,
    )
    print(
        manifest["status"],
        "worst acceptance fraction",
        max(item["max_error_ratio"] for item in manifest["diagnostics"]),
    )
    return 0 if all_ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--venv", type=Path)
    parser.add_argument("--readback", type=Path)
    options = parser.parse_args()
    if options.readback:
        readback(options.readback)
    else:
        if not all((options.output_dir, options.source_dir, options.venv)):
            parser.error("--output-dir, --source-dir, --venv are required")
        raise SystemExit(run(options))
