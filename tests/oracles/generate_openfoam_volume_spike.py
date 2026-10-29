"""OpenFOAM volume-oracle spike (Phase 0, L-OFSPIKE').

Synthetic-only spike (no real mesh required) that asks: is the 6-tetra hex
volume decomposition used by ``gragra.adapters.meshio_adapter.HEX_TETS_0_6``
(via ``from_meshio_volume``) accurate enough to serve as an A2-grade volume
*oracle* for OpenFOAM hexahedral cells?

This script does **not** re-derive the acceptance threshold -- it consumes
``tau_oracle = 1e-13`` and ``tau_affine = 1e-14`` as fixed inputs from
``docs/validation/tier_error_budget.md`` §5 (L-BUDGET), and re-uses the two
*independent* A2 oracles already validated in ``tests/_hex_volume_oracle.py``
(``hex_volume_24tetra``, ``hex_volume_gauss2``) unmodified, plus the
production 6-tetra corner-index table ``HEX_TETS_0_6`` unmodified.

Two required warnings from L-BUDGET §5.3, both honoured below:

1. Warp must be measured as **face non-planarity**
   (``eta_f = det(p_b-p_a, p_c-p_a, p_d-p_a) / h**3``), not as a naive
   "twist vector" magnitude -- a planar trapezoidal-prism warp has a large
   twist-vector norm but exactly zero 6-tetra error.
2. A single deterministic warp knob can cancel by symmetry (an observed
   example: a top-face rotation warp with eta/h ~= 0.1 giving 6-tetra error
   2.4e-16). The sweep below therefore perturbs all 8 corners with an
   *ensemble* of independent random draws per warp level and judges
   acceptance on the ensemble **maximum** error, never a single sample.

Steps performed:

(a) Affine pre-check (``tau_affine``): random affine (shear + anisotropic
    scale) hexahedra have every face exactly planar (affine maps preserve
    planarity), so 6-tetra and 24-tetra must agree to float64 rounding.
    This validates corner-order agreement between ``HEX_TETS_0_6``
    (meshio/VTK order) and ``_hex_volume_oracle.py`` (SPECFEM
    ``usual_hex_nodes`` order) *before* attributing any measured
    discrepancy to the 6-tetra algorithm itself.
(b) Warp sweep: for each of several warp levels ``w`` (random per-corner
    displacement magnitude, relative to the cube edge length ``h``), draw
    an ensemble of random 8-corner perturbations, measure the realised
    face-non-planarity ``eta`` and the relative volume error of 6-tetra
    (and, for reference, Gauss) against the 24-tetra reference.
(c) Summarise the ensemble-maximum error as a function of ``eta`` (JSON,
    plot-ready) and compare against ``tau_oracle`` to render an
    accept/reject verdict, including the realistic-mesh-warp margin
    (``eta ~ 1e-4``, L-BUDGET's working estimate).
(d) Write everything to ``--out`` (a non-committed path, e.g. ``/tmp``).

Usage::

    PYTHONPATH=tests conda run -n gragra python \\
        tests/oracles/generate_openfoam_volume_spike.py --out /tmp/of_spike.json
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import numpy as np

# Independent A2 oracles (read-only re-use, per plan write-scope).
from _hex_volume_oracle import hex_volume_24tetra, hex_volume_gauss2

# Production 6-tetra corner-index table (read-only re-use).
from gragra.adapters.meshio_adapter import HEX_TETS_0_6

# --------------------------------------------------------------------------
# Thresholds handed down by L-BUDGET (docs/validation/tier_error_budget.md
# §5.1 / §5.2 / §5.5) -- fixed inputs, not re-derived here.
# --------------------------------------------------------------------------
TAU_ORACLE = 1e-13
TAU_AFFINE = 1e-14
ETA_TYPICAL_REAL_MESH = 1e-4  # L-BUDGET's working estimate for real OpenFOAM meshes

# Reference (unit) hexahedron corners, meshio/VTK_HEXAHEDRON order -- the
# same corner order as both HEX_TETS_0_6 and tests/_hex_volume_oracle.py's
# usual_hex_nodes table (confirmed identical per tier_error_budget.md §5.5).
_REF_CORNERS = np.array(
    [
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
    ],
    dtype=np.float64,
)
_EDGE_LENGTH_H = 2.0  # edge length of _REF_CORNERS (cube spans [-1, 1])

# The 6 faces as outward-wound 4-tuples of local corner indices (same table
# as tests/_hex_volume_oracle.py's _HEX_FACES; re-declared locally rather
# than importing a private symbol from a file we must not modify).
_HEX_FACES: tuple[tuple[int, int, int, int], ...] = (
    (0, 3, 2, 1),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
)


def hex_volume_6tetra(corners8: np.ndarray) -> float:
    """6-tetrahedron decomposition volume, replicating the production
    formula in ``gragra.adapters.meshio_adapter.from_meshio_volume``
    (including its per-tetra ``abs()``), using the unmodified
    ``HEX_TETS_0_6`` corner-index table.

    This is the *candidate* method under evaluation -- not an oracle.
    """
    corners = np.asarray(corners8, dtype=np.float64)
    tet_nodes = corners[HEX_TETS_0_6]  # (6, 4, 3)
    a, b, c, d = tet_nodes[:, 0], tet_nodes[:, 1], tet_nodes[:, 2], tet_nodes[:, 3]
    u = b - a
    v = c - a
    w = d - a
    cross = np.cross(v, w)
    det = np.sum(u * cross, axis=-1)  # (6,)
    vol_t = np.abs(det) / 6.0
    return float(np.sum(vol_t))


def face_nonplanarity(corners8: np.ndarray, h: float = _EDGE_LENGTH_H) -> float:
    """eta = max over the 6 faces of |det(p_b-p_a, p_c-p_a, p_d-p_a)| / h**3.

    eta == 0 iff every face is planar (per tier_error_budget.md §5.3);
    this is the warp metric that actually governs 6-tetra error, not a
    naive corner-displacement / twist-vector norm.
    """
    corners = np.asarray(corners8, dtype=np.float64)
    eta_max = 0.0
    for a_idx, b_idx, c_idx, d_idx in _HEX_FACES:
        pa, pb, pc, pd = corners[a_idx], corners[b_idx], corners[c_idx], corners[d_idx]
        eta_f = abs(np.dot(pb - pa, np.cross(pc - pa, pd - pa))) / h**3
        eta_max = max(eta_max, eta_f)
    return eta_max


def _relative_diff(candidate: float, reference: float) -> float:
    """Relative difference guarded against division by zero (numerical.md)."""
    denom = abs(reference)
    if denom == 0.0:
        raise ValueError(
            "reference volume is exactly zero -- cannot form a relative "
            "difference (degenerate hexahedron?)"
        )
    return abs(candidate - reference) / denom


# --------------------------------------------------------------------------
# (a) Affine pre-check
# --------------------------------------------------------------------------


def _random_affine_matrix(rng: np.random.Generator) -> np.ndarray:
    """A well-conditioned, orientation-preserving affine matrix: positive
    diagonal scale (0.5..2.0) plus moderate shear (|off-diag| <= 0.4).

    Retries if the determinant is not comfortably positive, so the mapped
    hexahedron never inverts (which would otherwise flip the sign
    convention of hex_volume_24tetra and make the "affine => exact
    agreement" pre-check meaningless).
    """
    for _ in range(50):
        diag = rng.uniform(0.5, 2.0, size=3)
        shear = rng.uniform(-0.4, 0.4, size=(3, 3))
        np.fill_diagonal(shear, 0.0)
        matrix = np.diag(diag) + shear
        if np.linalg.det(matrix) > 0.1:  # comfortably orientation-preserving
            return matrix
    raise RuntimeError("failed to draw a well-conditioned affine matrix")


def run_affine_precheck(rng: np.random.Generator, n_trials: int = 20) -> dict:
    """Step (a): affine (non-warp) hexahedra must have eta == 0 (up to
    float64 rounding) and 6-tetra must match 24-tetra within tau_affine.

    This isolates corner-order agreement from the warp-error study proper
    (per docs/plans/... §5 L-OFSPIKE' step (a) / tier_error_budget.md §5.5).
    """
    trials = []
    for i in range(n_trials):
        matrix = _random_affine_matrix(rng)
        translation = rng.uniform(-10.0, 10.0, size=3)
        corners = _REF_CORNERS @ matrix.T + translation

        eta = face_nonplanarity(corners)
        vol_24 = hex_volume_24tetra(corners)
        vol_gauss = hex_volume_gauss2(corners)
        vol_6 = hex_volume_6tetra(corners)

        diff_6 = _relative_diff(vol_6, vol_24)
        diff_gauss = _relative_diff(vol_gauss, vol_24)

        trials.append(
            {
                "trial": i,
                "eta": eta,
                "volume_24tetra_m3": vol_24,
                "rel_diff_6tetra_vs_24tetra": diff_6,
                "rel_diff_gauss_vs_24tetra": diff_gauss,
            }
        )

    max_diff_6 = max(t["rel_diff_6tetra_vs_24tetra"] for t in trials)
    max_diff_gauss = max(t["rel_diff_gauss_vs_24tetra"] for t in trials)
    max_eta = max(t["eta"] for t in trials)

    passed = max_diff_6 < TAU_AFFINE
    return {
        "n_trials": n_trials,
        "tau_affine": TAU_AFFINE,
        "max_eta_over_trials": max_eta,
        "max_rel_diff_6tetra_vs_24tetra": max_diff_6,
        "max_rel_diff_gauss_vs_24tetra": max_diff_gauss,
        "pass": passed,
        "trials": trials,
    }


# --------------------------------------------------------------------------
# (b)/(c) Warp ensemble sweep
# --------------------------------------------------------------------------

#: Warp levels: per-corner random displacement magnitude relative to h,
#: spanning near-affine (1e-9) up to strongly distorted (1e-1). Chosen as a
#: decade sweep so the realistic-mesh estimate (eta ~ 1e-4, see
#: ETA_TYPICAL_REAL_MESH) is measured directly rather than extrapolated.
_WARP_LEVELS_W: tuple[float, ...] = (
    1e-9,
    1e-8,
    1e-7,
    1e-6,
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    1e-1,
)
_N_SAMPLES_PER_LEVEL = 12  # >= 8 per L-BUDGET §5.3's "8 samples or more"


def _random_unit_vectors(rng: np.random.Generator, n: int) -> np.ndarray:
    """n random unit vectors in R^3 (uniform on the sphere via normalized
    Gaussian draws)."""
    v = rng.normal(size=(n, 3))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms


def _perturbed_hex(rng: np.random.Generator, w_target: float) -> np.ndarray:
    """Perturb all 8 corners of the reference cube independently by a
    random vector of magnitude w_target * h in a random direction."""
    directions = _random_unit_vectors(rng, 8)
    displacement = w_target * _EDGE_LENGTH_H * directions
    return _REF_CORNERS + displacement


def run_warp_sweep(base_seed: int) -> dict:
    """Step (b)/(c): for each warp level, draw an ensemble of independent
    random 8-corner perturbations and record the *ensemble maximum*
    relative error of 6-tetra (and Gauss) against 24-tetra, as a function
    of the measured face-non-planarity eta.

    Per L-BUDGET §5.3's warning, acceptance must never be judged from a
    single deterministic sample -- only the ensemble maximum is used below.
    """
    levels = []
    all_samples = []  # pooled (eta, err_6tetra, err_gauss) for a global fit

    for level_idx, w_target in enumerate(_WARP_LEVELS_W):
        samples = []
        for sample_idx in range(_N_SAMPLES_PER_LEVEL):
            # Distinct, reproducible seed per (level, sample).
            seed = base_seed + 1000 * level_idx + sample_idx
            rng = np.random.default_rng(seed)
            corners = _perturbed_hex(rng, w_target)

            eta = face_nonplanarity(corners)
            vol_24 = hex_volume_24tetra(corners)
            vol_gauss = hex_volume_gauss2(corners)
            vol_6 = hex_volume_6tetra(corners)

            err_6 = _relative_diff(vol_6, vol_24)
            err_gauss = _relative_diff(vol_gauss, vol_24)

            sample = {
                "seed": seed,
                "eta": eta,
                "err_6tetra": err_6,
                "err_gauss": err_gauss,
            }
            samples.append(sample)
            all_samples.append({"w_target": w_target, **sample})

        etas = [s["eta"] for s in samples]
        err6s = [s["err_6tetra"] for s in samples]
        err_gausses = [s["err_gauss"] for s in samples]

        levels.append(
            {
                "w_target": w_target,
                "n_samples": _N_SAMPLES_PER_LEVEL,
                "eta_min": min(etas),
                "eta_max": max(etas),
                "eta_mean": float(np.mean(etas)),
                "err_6tetra_max": max(err6s),
                "err_6tetra_mean": float(np.mean(err6s)),
                "err_gauss_max": max(err_gausses),
                "err_gauss_mean": float(np.mean(err_gausses)),
                # eta/err pairs sorted for direct plotting (max-error curve).
                "samples": samples,
            }
        )

    return {
        "base_seed": base_seed,
        "n_levels": len(_WARP_LEVELS_W),
        "n_samples_per_level": _N_SAMPLES_PER_LEVEL,
        "levels": levels,
        "all_samples": all_samples,
    }


def _fit_power_law(eta_values: list[float], err_values: list[float]) -> dict | None:
    """Fit err ~= coeff * eta**exponent via a log-log least squares line.

    Returns None if fewer than 2 strictly-positive (eta, err) pairs are
    available (e.g. degenerate all-zero-error sweep).
    """
    pairs = [
        (e, v) for e, v in zip(eta_values, err_values, strict=True) if e > 0 and v > 0
    ]
    if len(pairs) < 2:
        return None
    log_eta = np.log10([p[0] for p in pairs])
    log_err = np.log10([p[1] for p in pairs])
    slope, intercept = np.polyfit(log_eta, log_err, 1)
    coeff = float(10.0**intercept)
    return {"exponent": float(slope), "coefficient": coeff, "n_points": len(pairs)}


def render_verdict(warp_sweep: dict) -> dict:
    """Step (c): compare the ensemble-maximum error curve to tau_oracle and
    render an accept/reject verdict for 6-tetra as an A2-grade oracle,
    including the realistic-mesh-warp margin.
    """
    levels = warp_sweep["levels"]

    # Largest swept level whose ensemble-maximum 6-tetra error still stays
    # under tau_oracle (None if even the smallest swept level fails).
    within_tolerance = [lvl for lvl in levels if lvl["err_6tetra_max"] < TAU_ORACLE]
    measured_eta_max_within_tolerance = (
        max(lvl["eta_max"] for lvl in within_tolerance) if within_tolerance else None
    )

    all_samples = warp_sweep["all_samples"]
    fit = _fit_power_law(
        [s["eta"] for s in all_samples], [s["err_6tetra"] for s in all_samples]
    )
    predicted_eta_max_from_fit = None
    if fit is not None and fit["coefficient"] > 0:
        # tau_oracle = coeff * eta**exponent  =>  eta = (tau_oracle/coeff)**(1/exponent)
        predicted_eta_max_from_fit = float(
            (TAU_ORACLE / fit["coefficient"]) ** (1.0 / fit["exponent"])
        )

    # Overall acceptance: 6-tetra qualifies as an A2-grade oracle only if
    # its ensemble-max error stays under tau_oracle across the *entire*
    # swept warp range (i.e. even the mildest non-affine warp we tested).
    accepted_as_a2_oracle = all(lvl["err_6tetra_max"] < TAU_ORACLE for lvl in levels)

    # Realistic-mesh margin: nearest measured level to eta_typical, judged
    # on a log scale (the sweep spans 8 decades, so a linear-distance
    # comparison would spuriously prefer far-smaller-eta levels).
    nearest_level = min(
        levels,
        key=lambda lvl: abs(
            np.log10(lvl["eta_mean"]) - np.log10(ETA_TYPICAL_REAL_MESH)
        ),
    )
    margin_ratio = (
        nearest_level["err_6tetra_max"] / TAU_ORACLE if TAU_ORACLE > 0 else float("inf")
    )
    margin_orders_of_magnitude = (
        float(np.log10(margin_ratio)) if margin_ratio > 0 else float("-inf")
    )

    return {
        "tau_oracle": TAU_ORACLE,
        "accepted_as_a2_oracle": accepted_as_a2_oracle,
        "measured_eta_max_within_tolerance": measured_eta_max_within_tolerance,
        "power_law_fit": fit,
        "predicted_eta_max_from_fit": predicted_eta_max_from_fit,
        "realistic_mesh_margin": {
            "eta_typical_real_mesh": ETA_TYPICAL_REAL_MESH,
            "nearest_measured_level_w_target": nearest_level["w_target"],
            "nearest_measured_level_eta_mean": nearest_level["eta_mean"],
            "err_6tetra_max_at_nearest_level": nearest_level["err_6tetra_max"],
            "margin_ratio_error_over_tau_oracle": margin_ratio,
            "margin_orders_of_magnitude": margin_orders_of_magnitude,
        },
        "rationale": (
            "6-tetra relative error grows ~linearly with the face "
            "non-planarity metric eta (fit exponent above); tau_oracle is "
            "exceeded already at the smallest tested non-affine warp "
            "levels, so 6-tetra is rejected as an A2-grade oracle. This "
            "independently reproduces docs/design/specfem_import.md's "
            "existing exclusion of 6-tetra from A2."
            if not accepted_as_a2_oracle
            else "6-tetra stayed under tau_oracle across the full swept warp range."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="output JSON path (non-committed, e.g. /tmp/of_spike.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260804,
        help="base RNG seed (reproducibility)",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    affine_precheck = run_affine_precheck(rng)
    if not affine_precheck["pass"]:
        # Fail fast: a corner-order mismatch would otherwise masquerade as
        # a false "6-tetra warp error" in the sweep below (per
        # tier_error_budget.md §5.5 -- this is exactly what step (a) exists
        # to catch before it is mis-attributed).
        raise AssertionError(
            "affine pre-check FAILED: max rel diff "
            f"{affine_precheck['max_rel_diff_6tetra_vs_24tetra']:.3e} >= "
            f"tau_affine={TAU_AFFINE:.3e}. This indicates a corner-order "
            "mismatch between HEX_TETS_0_6 (meshio/VTK order) and "
            "_hex_volume_oracle.py (usual_hex_nodes order), NOT warp "
            "error -- do not proceed to the warp sweep until resolved."
        )

    warp_sweep = run_warp_sweep(base_seed=args.seed)
    verdict = render_verdict(warp_sweep)

    result = {
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "description": (
            "Synthetic (no real mesh) OpenFOAM volume-oracle spike "
            "(Phase 0 L-OFSPIKE'). Evaluates whether the 6-tetra hex "
            "volume decomposition (HEX_TETS_0_6) qualifies as an "
            "A2-grade oracle, using thresholds handed down by L-BUDGET."
        ),
        "thresholds": {"tau_oracle": TAU_ORACLE, "tau_affine": TAU_AFFINE},
        "affine_precheck": affine_precheck,
        "warp_sweep": warp_sweep,
        "verdict": verdict,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=False))

    # Human-readable summary on stdout.
    print(
        f"affine precheck: pass={affine_precheck['pass']} "
        f"(max rel diff {affine_precheck['max_rel_diff_6tetra_vs_24tetra']:.3e} "
        f"< tau_affine={TAU_AFFINE:.3e})"
    )
    print(f"6-tetra accepted as A2-grade oracle: {verdict['accepted_as_a2_oracle']}")
    if verdict["power_law_fit"] is not None:
        fit = verdict["power_law_fit"]
        print(
            f"power-law fit: err_6tetra ~= {fit['coefficient']:.3e} * "
            f"eta^{fit['exponent']:.4f}  (n={fit['n_points']})"
        )
    margin = verdict["realistic_mesh_margin"]
    print(
        f"realistic-mesh margin @ eta~{margin['eta_typical_real_mesh']:.0e}: "
        f"err_6tetra_max={margin['err_6tetra_max_at_nearest_level']:.3e} is "
        f"{margin['margin_ratio_error_over_tau_oracle']:.3e}x tau_oracle "
        f"({margin['margin_orders_of_magnitude']:.1f} orders of magnitude over)"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
