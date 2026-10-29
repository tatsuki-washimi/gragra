"""Write immutable U10 finite-domain validation evidence and read it back.

Run from the dedicated installed worktree. All heavy invocations must be wrapped
by the shared flock and single-thread environment described in the design.
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

# Explicit worktree root, never another checkout or an external PYTHONPATH.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from gragra.analytic import u10_spherical_shell_general as production  # noqa: E402
from tests.oracles.u10_spherical_shell_general_reference import reference  # noqa: E402

_PARTS = ("bulk", "inner_wall", "outer_boundary", "total")
_NAMES = ("potential", "acceleration", "gradient")
_SLICES = (slice(0, 1), slice(1, 4), slice(4, 13))
_SETTINGS = ((48, 24, 96), (64, 32, 128), (96, 48, 192))


def fixtures():
    direction = np.array([1.0, 2.0, 3.0])
    axis = np.array([2.0, -3.0, 1.0])
    axis /= np.linalg.norm(axis)
    center = np.array([0.3, -0.5, 0.2])
    for ratio in (0.2, 0.8, 0.99, 0.9999):
        for kb in (0.01, 0.8, 5.0):
            for wave in ("P", "S"):
                a = ratio * 10
                targets = (
                    np.array(
                        [np.zeros(3), 0.4 * a * axis, (a + 10) / 2 * axis, 20 * axis]
                    )
                    + center
                )
                kw = {
                    "inner_radius_m": a,
                    "outer_radius_m": 10.0,
                    "density_kg_m3": 2300.0,
                    "wavenumber_rad_m": kb / 10,
                    "displacement_m": 1e-9 * (1 + 0.4j),
                    "propagation_direction": direction.tolist(),
                    "polarization": direction.tolist()
                    if wave == "P"
                    else [2.0, -1.0, 0.0],
                    "wave_type": wave,
                    "center_m": center.tolist(),
                    "phase_origin_m": [-0.7, 0.2, 0.9],
                }
                yield f"{wave}-a{ratio}-kb{kb}", targets, kw


def _dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _identity():
    import gragra

    expected = Path(sys.prefix).resolve()
    assert sys.prefix != sys.base_prefix, "use a dedicated virtual environment"
    assert Path(sys.executable).parent.parent.resolve() == expected
    assert Path(gragra.__file__).resolve().is_relative_to(_ROOT)
    assert Path(production.__file__).resolve().is_relative_to(_ROOT)
    for path in sys.path:
        assert "/.paseo/worktrees/" not in path or str(_ROOT) in path, path
        candidate = Path(path) / "gragra" / "__init__.py"
        if candidate.is_file():
            assert candidate.resolve().is_relative_to(_ROOT), path
    return {
        "executable": sys.executable,
        "sys_path": sys.path,
        "gragra_file": gragra.__file__,
        "module_file": production.__file__,
        "python": sys.version,
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "mpmath", "sympy", "pytest")
        },
        "longdouble_epsilon": float(np.finfo(np.longdouble).eps),
    }


def readback(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    for file, expected in manifest["hashes"].items():
        assert (
            hashlib.sha256((directory / file).read_bytes()).hexdigest() == expected
        ), file
    with np.load(directory / "raw.npz", allow_pickle=False) as arrays:
        assert sorted(arrays.files) == sorted(manifest["arrays"])
        for name, expected in manifest["arrays"].items():
            arr = arrays[name]
            assert list(arr.shape) == expected["shape"]
            assert str(arr.dtype) == expected["dtype"]
            assert np.isfinite(arr).all()
            assert hashlib.sha256(arr.tobytes()).hexdigest() == expected["sha256"]
    print(
        f"READBACK_PASS {len(manifest['arrays'])} arrays; hashes, names, "
        "shapes, dtypes, finite values and bytes"
    )


def validate(directory):
    environment = _identity()
    directory.mkdir(parents=True, exist_ok=False)
    arrays = {}
    records = []
    cases = []
    production_diagnostics = {}
    reference_diagnostics = {}
    _dump(directory / "environment.json", environment)
    for name, targets, kw in fixtures():
        print(name, flush=True)
        amplitude = kw["displacement_m"]
        cases.append(
            {
                "name": name,
                "parameters": {
                    **kw,
                    "displacement_m": [amplitude.real, amplitude.imag],
                },
                "targets_m": targets.tolist(),
                "units": {
                    "targets": "m",
                    "radii": "m",
                    "density": "kg/m^3",
                    "k": "rad/m",
                    "U": "m",
                },
            }
        )
        arrays[name + "-targets"] = targets
        results = []
        try:
            fields, diag = production.spherical_shell_observables_g_free(
                targets, return_diagnostics=True, **kw
            )
            production_diagnostics[name] = diag
            for part in _PARTS:
                arrays[name + "-production-" + part] = np.concatenate(
                    (
                        fields[part].potential[:, None],
                        fields[part].acceleration,
                        fields[part].gradient.reshape(-1, 9),
                    ),
                    axis=1,
                )
            fine_fields, fine_diag = production.spherical_shell_observables_g_free(
                targets,
                return_diagnostics=True,
                rtol=np.array([1e-10, 1e-10, 1e-9]) / 16,
                atol=4
                * np.pi
                * 2300
                * abs(amplitude)
                * np.array([10, 1, 0.1])
                * (10 - kw["inner_radius_m"])
                / 10
                * 1e-15,
                max_degree=192,
                **kw,
            )
            production_diagnostics[name + "-refined"] = fine_diag
            for part in _PARTS:
                arrays[name + "-refined-" + part] = np.concatenate(
                    (
                        fine_fields[part].potential[:, None],
                        fine_fields[part].acceleration,
                        fine_fields[part].gradient.reshape(-1, 9),
                    ),
                    axis=1,
                )
            for level, (nphi, nradial, nmu) in enumerate(_SETTINGS):
                ref, diagnostics = reference(
                    targets, **kw, nphi=nphi, nradial=nradial, nmu=nmu
                )
                reference_diagnostics[name + f"-{level}"] = diagnostics
                results.append(ref)
                for part in _PARTS:
                    arrays[name + f"-reference{level}-" + part] = ref[part]
        except (ValueError, RuntimeError, ArithmeticError) as exc:
            records.append(
                {
                    "case": name,
                    "status": "REFERENCE_NOT_CONVERGED"
                    if "REFERENCE" in str(exc)
                    else "IMPLEMENTATION_FAILURE",
                    "error": str(exc),
                }
            )
            if hasattr(exc, "diagnostics"):
                production_diagnostics[name] = exc.diagnostics
            continue
        for part in _PARTS:
            for target in range(len(targets)):
                for observable, sl in enumerate(_SLICES):
                    scale = 4 * np.pi * 2300 * 1e-9 * [10, 1, 0.1][observable]
                    thickness_scaled = part in ("bulk", "total") and target != 2
                    if thickness_scaled:
                        scale *= (10 - kw["inner_radius_m"]) / 10
                    ref = results[-1][part][target, sl]
                    previous = results[-2][part][target, sl]
                    value = arrays[name + "-production-" + part][target, sl]
                    relative = [1e-10, 1e-10, 1e-9][observable]
                    budget = 1e-12 * scale + relative * np.linalg.norm(ref)
                    component_budget = 1e-12 * scale + relative * abs(ref)
                    difference = abs(value - ref)
                    refinement = abs(ref - previous)
                    degree_refinement = float(
                        np.linalg.norm(
                            value - arrays[name + "-refined-" + part][target, sl]
                        )
                    )
                    error = float(np.linalg.norm(value - ref))
                    refine = float(np.linalg.norm(ref - previous))
                    reference_ok = refine <= budget / 10 and np.all(
                        refinement <= component_budget / 10
                    )
                    implementation_ok = (
                        degree_refinement <= budget / 10
                        and error <= budget
                        and np.all(difference <= component_budget)
                    )
                    wall_sum = np.linalg.norm(
                        results[-1]["inner_wall"][target, sl]
                    ) + np.linalg.norm(results[-1]["outer_boundary"][target, sl])
                    records.append(
                        {
                            "case": name,
                            "target": target,
                            "part": part,
                            "observable": _NAMES[observable],
                            "status": "REFERENCE_NOT_CONVERGED"
                            if not reference_ok
                            else "PASS"
                            if implementation_ok
                            else "IMPLEMENTATION_FAILURE",
                            "scale": scale,
                            "thickness_scaled": thickness_scaled,
                            "acceptance": float(budget),
                            "implementation_error": error,
                            "production_refinement": degree_refinement,
                            "reference_refinement": refine,
                            "previous_refinement": float(
                                np.linalg.norm(previous - results[-3][part][target, sl])
                            ),
                            "component_error": difference.astype(float).tolist(),
                            "component_refinement": refinement.astype(float).tolist(),
                            "component_acceptance": component_budget.astype(
                                float
                            ).tolist(),
                            "wall_cancellation_ratio": float(
                                wall_sum / (scale + np.linalg.norm(ref))
                            ),
                            "model_approximation": "NOT_APPLICABLE",
                        }
                    )
    np.savez_compressed(directory / "raw.npz", **arrays)
    _dump(directory / "inputs.json", cases)
    _dump(directory / "diagnostics-production.json", production_diagnostics)
    _dump(directory / "diagnostics-reference.json", reference_diagnostics)
    _dump(directory / "results.json", records)
    source_root = (
        directory.parents[1]
        / "f665ee6c500001721966d161ea04ffdfff72f827"
        / "preflight"
        / "sources"
    )
    sources = (
        json.loads((source_root / "source-manifest.json").read_text())
        if (source_root / "source-manifest.json").exists()
        else "NOT_AVAILABLE"
    )
    manifest = {
        "code_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "dirty": subprocess.check_output(["git", "status", "--porcelain"], text=True),
        "command": sys.argv,
        "source_hashes": sources,
        "settings": _SETTINGS,
        "threads": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "units": ["kg/m", "kg/m^2", "kg/m^3"],
        "component_order": "potential, ax, ay, az, row-major gradient",
        "arrays": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
            }
            for name, value in arrays.items()
        },
        "hashes": {
            file.name: hashlib.sha256(file.read_bytes()).hexdigest()
            for file in directory.iterdir()
            if file.is_file()
        },
    }
    _dump(directory / "manifest.json", manifest)
    child = subprocess.run(
        [sys.executable, __file__, "--readback", str(directory)],
        capture_output=True,
        text=True,
    )
    (directory / "readback.log").write_text(child.stdout + child.stderr)
    assert child.returncode == 0, child.stderr
    counts = {
        status: sum(row["status"] == status for row in records)
        for status in ("PASS", "REFERENCE_NOT_CONVERGED", "IMPLEMENTATION_FAILURE")
    }
    print(json.dumps(counts))
    print(child.stdout)
    return (
        0
        if counts["REFERENCE_NOT_CONVERGED"] == counts["IMPLEMENTATION_FAILURE"] == 0
        else 1
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output-dir", type=Path)
    group.add_argument("--readback", type=Path)
    args = parser.parse_args()
    if args.readback:
        readback(args.readback)
    else:
        sys.exit(validate(args.output_dir))
