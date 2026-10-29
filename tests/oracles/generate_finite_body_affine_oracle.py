"""Independent Newton-gravity mass-displacement oracle for a globally-affine
displacement field over a uniform-gradient-density box.

This script computes an independent reference value for the "mass
displacement" (linearized dipole) gravity observable of
``gragra.fields.finite_body_p1.piecewise_tetra_p1_density_sources`` /
``direct_point_displacement_sum``, WITHOUT importing gragra at all -- it
shares no code with the production tetra/face quadrature
(``gragra.geometry.tetra_quadrature``) under test.

Why an affine field eliminates interpolation error
----------------------------------------------------
``piecewise_tetra_p1_density_sources`` represents density and displacement
as P1 (piecewise-linear) fields on a tetra mesh. A P1 basis interpolates any
GLOBALLY AFFINE field ``f(x) = f0 + grad_f . (x - c)`` *exactly*, at every
point, on *any* mesh -- there is no discretization/interpolation error to
separate from quadrature error. Choosing both the density ``rho(x)`` and
the displacement ``xi(x)`` to be globally affine over a single box domain
therefore isolates *quadrature order* as the only source of residual
between production's mass-displacement representation and the true
continuum integral -- exactly what
``src/gragra/geometry/tetra_quadrature.py``'s selectable higher/local order
is meant to test (see ``docs/research/wave4-6-F.md``, stage-(4)/(5), and
the "⑥-B" section this oracle supports).

Independent route
------------------
The continuum mass-displacement integral

    Q(target) = int_box rho(x) * K_disp(x, target, xi(x)) dV

(``K_disp`` is the same linearized-dipole kernel already independently
re-derived in ``tests/_finite_body_p1_reference.py``'s ``_kernel`` -- see
that module's own use as a continuum oracle in
``tests/test_finite_body_p1.py::test_fixed_refinement_and_independent_halfbox_oracle``)
has no elementary closed form for a general kernel (it is the "gravity of a
polyhedron with a linearly varying density times a linear source-weight"
problem), so this script evaluates it via its own from-scratch tensor
Gauss-Legendre quadrature over the box (an axis-aligned box, not a Duffy
tetra map, so it shares no coordinate transform with
``tetra_gauss_quadrature``/``triangle_gauss_quadrature``), at a sequence of
increasing orders, and records the convergence of that sequence as the
correctness evidence (analogous in spirit to
``generate_halfspace_oracle.py``'s R_cut extrapolation, but here a direct
Gauss order sweep suffices because the box-and-targets geometry has no
truncation parameter and the integrand is smooth and non-oscillatory well
away from the domain).

Output
------
Writes ``tests/oracles/data/finite_body_affine_v1.json``, a deterministic
function of this script's fixed constants and grids (G is *not* applied --
the recorded values are G-free, matching gragra's kernel-contract boundary;
``tests/test_finite_body_p1_quadrature.py`` multiplies by ``G_SI`` itself
before comparing against production's ``G_SI``-scaled observables, or
compares G-free to G-free -- see that test module).

This JSON *is* committed to git, following the same documented exception to
the project's "no generated result artifacts in git" convention used by
``tests/oracles/generate_halfspace_oracle.py`` / ``halfspace_v1.json``
(``.harness/rules/common/scope.md``): a versioned, provenance-carrying
golden fixture (generation script, units, sign convention, precision/order
stages, error estimates, generation-time commit SHA) that
``tests/test_finite_body_p1_quadrature.py`` reads as fixed input, not an
ephemeral analysis result. Provenance uses the same content-addressed
scheme as ``generate_halfspace_oracle.py`` (see that module's docstring for
the rationale): ``source_content_sha256``, ``derivation_input_sha256``,
``golden_artifact_sha256``; ``upstream_revision`` is attribution-only.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
OUTPUT_PATH = Path(__file__).parent / "data" / "finite_body_affine_v1.json"

ORACLE_NAME = "finite_body_affine_v1"
ORACLE_METHOD_VERSION = "1.0"
UPSTREAM_REPOSITORY = (
    "gragra (this repository) -- independent from-scratch tensor "
    "Gauss-Legendre box integral; does not import gragra or share code "
    "with gragra.geometry.tetra_quadrature"
)

# Proposed benchmark geometry/physics (meters, kg/m^3, kg/m^4, dimensionless
# strain-rate matrix entries) -- not a real-site measurement.
BOX_BOUNDS_M = ((-2.0, 2.0), (-1.5, 1.5), (-1.0, 1.0))
DENSITY_CENTER_M = (0.0, 0.0, 0.0)
DENSITY_RHO0_KG_M3 = 1800.0
DENSITY_GRADIENT_KG_M4 = (2.0, -1.5, 1.0)

DISPLACEMENT_CENTER_M = (0.0, 0.0, 0.0)
DISPLACEMENT_B_M = (0.02, -0.015, 0.01)
DISPLACEMENT_A = (
    (0.010, 0.002, -0.001),
    (-0.0015, 0.008, 0.001),
    (0.0005, -0.0012, 0.006),
)

TARGETS_M = (
    (5.0, 3.0, -3.0),
    (-4.0, -3.5, 2.5),
    (8.0, 0.0, 0.0),
)

OBSERVABLES = ("potential", "acceleration", "gradient")
ORDER_STAGES = (6, 8, 10, 12, 14, 16)


def density_kg_m3(points_m: np.ndarray) -> np.ndarray:
    """Globally-affine density: rho0 + grad_rho . (x - center)."""
    delta = points_m - np.asarray(DENSITY_CENTER_M)
    return DENSITY_RHO0_KG_M3 + delta @ np.asarray(DENSITY_GRADIENT_KG_M4)


def displacement_m(points_m: np.ndarray) -> np.ndarray:
    """Globally-affine displacement: b + A . (x - center)."""
    delta = points_m - np.asarray(DISPLACEMENT_CENTER_M)
    return np.asarray(DISPLACEMENT_B_M) + delta @ np.asarray(DISPLACEMENT_A).T


def _mass_displacement_kernel(
    points_m: np.ndarray, target_m: np.ndarray, u: np.ndarray, observable: str
) -> np.ndarray:
    """Linearized mass-displacement (dipole) kernel, G-free.

    Independently re-implements the same well-established linearization
    already used as a from-scratch continuum oracle in
    ``tests/_finite_body_p1_reference.py`` (its ``_kernel``): the
    first-order Taylor term of moving a point mass at ``points_m`` by
    ``u`` -- i.e. d/depsilon [K_static(points_m - epsilon*u, target_m)] at
    epsilon=0. Duplicated here (not imported) to keep this generator
    standalone and independent of the test package, exactly as
    ``generate_halfspace_oracle.py`` duplicates rather than imports.
    """
    d = points_m - target_m
    q = np.einsum("ni,ni->n", d, d)
    root = np.sqrt(q)
    z = np.einsum("ni,ni->n", d, u)
    if observable == "potential":
        return z / (q * root)
    if observable == "acceleration":
        return u / (q * root)[:, None] - 3.0 * d * (z / (q * q * root))[:, None]
    if observable == "gradient":
        eye = np.eye(3)
        term1 = (
            3.0
            * (
                np.einsum("ni,nj->nij", u, d)
                + np.einsum("ni,nj->nij", d, u)
                + eye[None, :, :] * z[:, None, None]
            )
            / (q * q * root)[:, None, None]
        )
        term2 = (
            15.0
            * np.einsum("ni,nj->nij", d, d)
            * z[:, None, None]
            / (q**3 * root)[:, None, None]
        )
        return term1 - term2
    raise ValueError(f"unknown observable {observable!r}")


def _gauss_1d(lo: float, hi: float, order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    x = 0.5 * (hi - lo) * nodes + 0.5 * (hi + lo)
    w = 0.5 * (hi - lo) * weights
    return x, w


def _box_quadrature_points(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Tensor Gauss-Legendre points/weights over ``BOX_BOUNDS_M``.

    Independent of ``gragra.geometry.tetra_quadrature``'s Duffy tetra/
    triangle map: this is a plain axis-aligned box tensor rule, sharing no
    coordinate transform or code with the production rule under test.
    """
    (x0, x1), (y0, y1), (z0, z1) = BOX_BOUNDS_M
    xs, wx = _gauss_1d(x0, x1, order)
    ys, wy = _gauss_1d(y0, y1, order)
    zs, wz = _gauss_1d(z0, z1, order)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    wxx, wyy, wzz = np.meshgrid(wx, wy, wz, indexing="ij")
    points = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3)
    weights = (wxx * wyy * wzz).reshape(-1)
    return points, weights


def mass_displacement_value(
    target_m: np.ndarray, observable: str, order: int
) -> np.ndarray:
    """G-free mass-displacement response at one target, one Gauss order."""
    points, weights = _box_quadrature_points(order)
    rho = density_kg_m3(points)
    u = displacement_m(points)
    kernel = _mass_displacement_kernel(points, target_m, u, observable)
    weight_shape = (len(weights),) + (1,) * (kernel.ndim - 1)
    return np.sum(
        weights.reshape(weight_shape) * rho.reshape(weight_shape) * kernel, axis=0
    )


def _array_record(value: np.ndarray) -> dict:
    array = np.asarray(value, dtype=np.float64)
    return {"shape": list(array.shape), "values": array.tolist()}


def _git_head_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def generator_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _derivation_input() -> dict:
    return {
        "box_bounds_m": [list(bound) for bound in BOX_BOUNDS_M],
        "density_center_m": list(DENSITY_CENTER_M),
        "density_rho0_kg_m3": DENSITY_RHO0_KG_M3,
        "density_gradient_kg_m4": list(DENSITY_GRADIENT_KG_M4),
        "displacement_center_m": list(DISPLACEMENT_CENTER_M),
        "displacement_b_m": list(DISPLACEMENT_B_M),
        "displacement_a": [list(row) for row in DISPLACEMENT_A],
        "targets_m": [list(target) for target in TARGETS_M],
        "order_stages": list(ORDER_STAGES),
    }


def derivation_input_sha256() -> str:
    encoded = json.dumps(_derivation_input(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def golden_artifact_sha256(non_provenance_body: dict) -> str:
    encoded = json.dumps(non_provenance_body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def generate() -> dict:
    targets = [np.asarray(target, dtype=np.float64) for target in TARGETS_M]
    points_out = []
    for target_index, target in enumerate(targets):
        observables_out = {}
        for observable in OBSERVABLES:
            values_by_order = {
                str(order): mass_displacement_value(target, observable, order)
                for order in ORDER_STAGES
            }
            finest = values_by_order[str(ORDER_STAGES[-1])]
            consecutive_diff = []
            for a, b in zip(ORDER_STAGES[:-1], ORDER_STAGES[1:], strict=True):
                diff = float(
                    np.linalg.norm(
                        np.ravel(values_by_order[str(b)] - values_by_order[str(a)])
                    )
                )
                consecutive_diff.append(diff)
            normalized_diff_vs_finest = [
                float(
                    np.linalg.norm(np.ravel(values_by_order[str(order)] - finest))
                    / max(float(np.linalg.norm(np.ravel(finest))), np.finfo(float).tiny)
                )
                for order in ORDER_STAGES
            ]
            observables_out[observable] = {
                "values_by_order": {
                    order: _array_record(value)
                    for order, value in values_by_order.items()
                },
                "consecutive_order_abs_diff": consecutive_diff,
                "normalized_diff_vs_finest_order": normalized_diff_vs_finest,
                "accepted_value": _array_record(finest),
                "accepted_order": ORDER_STAGES[-1],
            }
        points_out.append(
            {
                "target_index": target_index,
                "target_m": target.tolist(),
                "observables": observables_out,
            }
        )

    body = {
        "schema_version": SCHEMA_VERSION,
        "description": (
            "Independent from-scratch tensor Gauss-Legendre mass-displacement "
            "gravity oracle for a globally-affine density/displacement box, "
            "G-free. See the generator module docstring for the full "
            "derivation, independence, and convergence-evidence contract."
        ),
        "definition": {
            "domain": "axis-aligned box, BOX_BOUNDS_M",
            "density": "rho0 + grad_rho . (x - density_center_m)",
            "displacement": "b + A . (x - displacement_center_m)",
            "kernel": (
                "linearized mass-displacement (dipole) kernel, G-free; "
                "see _mass_displacement_kernel"
            ),
            "note": (
                "Values are G-free (no G_SI factor); multiply by G_SI before "
                "comparing to gragra's physical observables."
            ),
        },
        "order_stages": list(ORDER_STAGES),
        "constants": _derivation_input(),
        "points": points_out,
    }

    provenance = {
        "oracle_name": ORACLE_NAME,
        "oracle_method_version": ORACLE_METHOD_VERSION,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_revision": _git_head_sha(),
        "source_content_sha256": generator_sha256(),
        "derivation_input_sha256": derivation_input_sha256(),
        "golden_artifact_sha256": golden_artifact_sha256(body),
        "generator_script": "tests/oracles/generate_finite_body_affine_oracle.py",
        "independent_of": [
            "gragra.fields.finite_body_p1",
            "gragra.geometry.tetra_quadrature",
            "gragra.integrators",
        ],
        "convention_doc": "docs/research/wave4-6-F.md#6-b",
        "not_a_cross_validation_against_external_toolbox_execution": True,
    }

    return {
        "schema_version": body["schema_version"],
        "description": body["description"],
        "provenance": provenance,
        "definition": body["definition"],
        "order_stages": body["order_stages"],
        "constants": body["constants"],
        "points": body["points"],
    }


def main() -> None:
    data = generate()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    print(f"wrote {OUTPUT_PATH} ({len(data['points'])} points)")


if __name__ == "__main__":
    main()
