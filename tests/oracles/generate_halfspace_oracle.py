"""Independent Newton-gravity-integral oracle generator for the half-space
vertical-incidence P-wave transfer functions.

This script re-derives ``gragra.analytic.halfspace.halfspace_transfer_bulk`` /
``halfspace_transfer_full`` (``docs/architecture/halfspace_conventions.md``
Sec.5) through a route that shares no code with ``gragra`` -- it does NOT
import ``gragra.analytic`` or ``gragra.waves``. It computes the Newtonian
gravity of the wave-induced density perturbation via a 3D integral over
depth with the horizontal/radial direction reduced analytically through the
closed-form finite-disk field (not a literal triple integral), truncated in
both disk radius R_cut and vertical extent Z_max, independent of the
closed-form ``1 - 2 cos(kH)`` / ``-2 cos(kH)`` algebraic simplification used
in that module's implementation (ported from nn-sus, per conventions Sec.8).

Reference note: this compares against the *published closed-form formula*
(re-derived from first principles below), not against ANNA toolbox execution
output. See ``docs/validation/harms2019_coverage.md`` for the validation summary.

Physical setup (docs/architecture/halfspace_conventions.md Sec.5-6)
--------------------------------------------------------------------
A P wave is normally incident on a free surface at z=0 from within the
half-space z<=0 (z-up), with incident displacement amplitude ``xi_inc``
(time-harmonic, complex amplitude convention e^{-i*omega*t} implicit; all
quantities below use the real amplitude since normal incidence keeps
everything in phase). Free-surface reflection at normal incidence gives
R_PP = +1 (conventions Sec.6), so the total displacement field in the
half-space is a standing wave:

    u_z(z) = xi_inc * [e^{i k z} + e^{-i k z}] = 2 xi_inc cos(k z),   z <= 0

with k = omega / v_P. At the surface, u_z(0) = 2 xi_inc (the "surface
displacement multiplier = 2.0" of conventions Sec.6).

Two independent mass-perturbation channels contribute to the vertical
gravity response, and are kept SEPARATE throughout this script (both in the
per-cutoff-stage values and in the final extrapolated numbers), because
bulk/surface/total must be compared
individually rather than only as a total:

1. Bulk density perturbation (mass conservation to first order,
   ``delta_rho = -rho_0 * du_z/dz``):

       delta_rho(z') = 2 rho_0 xi_inc k sin(k z'),   z' <= 0

2. Free-surface mass sheet: the moving free surface exposes/covers a layer
   of thickness u_z(0) and density rho_0, i.e. an areal-density perturbation

       sigma_surface = rho_0 * u_z(0) = 2 rho_0 xi_inc

   located at z'=0.

Newtonian gravity of each channel is computed via the on-axis field of a
*finite disk* (not an infinite plane) of areal density sigma at height z',
observed at height z, radius R (elementary result from integrating rings
r in [0, R] with Newton's law and az = -G dm * d / (r^2+d^2)^(3/2)):

    a_z(d; R, sigma) = -2 pi G sigma * [ sign(d) - d / sqrt(d^2 + R^2) ],
    d = z_obs - z'

As R -> infinity this recovers the standard infinite-plane / Gauss's-law
result a_z -> -2 pi G sigma * sign(d) (the textbook basis of the closed-form
T_bulk/T_full expressions); at *finite* R it is the genuine truncated-volume
approximation this oracle integrates and then extrapolates R -> infinity
from, which is the independent numerical route used for this comparison (as
opposed to trusting the infinite-plane idealization algebraically).

Two truncations, extrapolated/checked separately
-------------------------------------------------
- **Horizontal radius R_cut**: the disk radius above. This is the physical
  finite-volume truncation whose R_cut -> infinity limit is the quantity of
  interest. At least 3 stages are used (this script uses 7:
  10, 20, 40, 80, 160, 320, 640, in the same length units as 1/k, with k
  fixed to 1.0 throughout -- H is then measured in the same units, so H = kH
  for every point).
- **Vertical numerical-quadrature limit Z_max**: for any *finite* R_cut, the
  disk-formula bracket [sign(d) - d/sqrt(d^2+R_cut^2)] decays like
  ``R_cut^2 / d^2`` for |d| >> R_cut, so the bulk integral over z' in
  (-infinity, 0] is absolutely convergent for any finite R_cut (no ad hoc
  regularization needed, unlike the R_cut -> infinity / infinite-plane limit
  taken directly, which is only conditionally convergent). ``Z_MAX_FACTOR =
  80`` (i.e. Z_max = 80*R_cut) is used for the main R_cut-stage sweep. This
  choice is re-verified reproducibly (not just via an undocumented scratch
  script) by :func:`z_max_sensitivity`, which re-evaluates the bulk integral
  at the largest R_cut stage for Z_MAX_FACTOR in (40, 80, 160) at every kH
  grid point and records the resulting ``tail_error_estimate`` (the gap
  between the two largest factors) in the output JSON;
  ``tests/test_halfspace_gravity_oracle.py`` checks this is small.
  The bulk integrand also has a known non-smooth point at z'=-h (the disk's
  own plane, where the ``sign(d)`` term in :func:`disk_az` jumps); this is
  passed to ``scipy.integrate.quad`` as an explicit ``points=[-h]``
  breakpoint (mirroring the breakpoints already used on the mpmath path
  below) rather than relying on adaptive subdivision to locate it.

Extrapolation contract (fixed BEFORE looking at any closed-form target)
-------------------------------------------------------------------------
For bulk, sheet, and full (=bulk+sheet) separately, per kH point: evaluate at
all R_cut stages, fit ``f(R) = c0 + c1/R + c2/R^2`` by least squares (3 free
parameters against >=6 data points), using ``curve_fit`` with an initial
guess derived ONLY from the largest-R_cut computed value (c0_guess =
f(R_max), c1_guess = c2_guess = 0) -- the closed-form target value is never
used as a fit input, only afterward as the pass/fail comparison in
``tests/test_halfspace_gravity_oracle.py``. ``c0`` is the reported
R_cut -> infinity extrapolated oracle value.

Fit-stability is itself re-verified rather than assumed: for all three
channels (bulk, sheet, full), the same fit is additionally run on the last
6 and last 5 R_cut stages (dropping the smallest 1 and 2 stages
respectively). The largest pairwise difference among the 7/6/5-stage ``c0``
values is recorded as ``fit_stability_max_diff`` per point/channel; a
stable extrapolation should barely move as the smallest (least-converged)
stages are dropped.

mpmath precision cross-check
------------------------------
For a small representative subset of (kH, R_cut) points (not the full grid,
to keep runtime bounded -- mpmath's oscillatory quadrature is 100-1000x
slower than scipy's float64 QUADPACK), the same bulk integral is
re-evaluated with ``mpmath.quad`` at two working precisions (dps=30 and
dps=50), with explicit half-period breakpoints supplied (mpmath's ``quad``
does not auto-subdivide a single huge interval to resolve oscillation the
way scipy's adaptive QUADPACK does). Agreement between dps=30, dps=50, and
the scipy float64 value confirms the scipy result is not an artifact of
floating-point rounding.

Output
------
Writes ``tests/oracles/data/halfspace_v1.json``, a deterministic function of
this script's fixed grids and constants (G=1, rho_0=1, xi_inc=1 -- see
``dimensionless_units`` in the output; T is dimensionless so the choice of
units is irrelevant to the comparison, only the constants must be recorded).

This JSON *is* committed to git -- an intentional, documented exception to
the project's general "no generated result artifacts in git" convention
(``CLAUDE.md`` / ``.harness/rules/common/scope.md``, which target ephemeral
CSV/PNG/result-JSON analysis output). Unlike those, this file is a
versioned, provenance-carrying golden fixture (generation script, units,
sign convention, precision/cutoff stages, error estimates) that
``tests/test_halfspace_gravity_oracle.py`` reads as fixed input. Keeping the
generator and result separately makes the numerical calculation reproducible.

Provenance recorded in the output is content-addressed (see
``tests/_provenance_contract.py`` module docstring
for why the earlier git-history-dependent design was replaced):
``source_content_sha256`` (a hash of this script's own source),
``derivation_input_sha256`` (a hash of the controlling constants/grids, see
``_derivation_input()``), and ``golden_artifact_sha256`` (a hash of the
generated data itself, see ``golden_artifact_sha256()``). ``upstream_revision``
(the git HEAD SHA at generation time) is recorded for human-readable
attribution only and is not verified by the standard provenance checks.
Regeneration procedure: after any change to this script or its constants,
rerun this script and commit the regenerated JSON -- the fail-closed checks
in ``tests/test_halfspace_gravity_oracle.py`` /
``tests/test_validation_extra_available.py`` catch a stale JSON regardless
of whether that order is followed, since they recompute all three hashes
fresh from the on-disk script and the on-disk JSON.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy import integrate
from scipy.optimize import curve_fit

G = 1.0
RHO_0 = 1.0
XI_INC = 1.0
K = 1.0  # fixed; H = kH / K = kH in these units

R_CUT_STAGES = (10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0)
Z_MAX_FACTOR = 80.0  # Z_max = Z_MAX_FACTOR * R_cut
QUAD_LIMIT = 20000

# Vertical (Z_max) truncation sensitivity: re-evaluated at the largest R_cut
# stage for each factor below; the gap between the two largest factors is
# recorded as the tail-truncation error estimate (see module docstring).
Z_MAX_SENSITIVITY_FACTORS = (40.0, 80.0, 160.0)
Z_MAX_SENSITIVITY_R_CUT = R_CUT_STAGES[-1]

# Cutoff-fit stability: extrapolation is re-run dropping the smallest 1 and 2
# R_cut stages; the largest pairwise difference among the resulting c0
# values is recorded as fit_stability_max_diff (see module docstring).
FIT_STABILITY_STAGE_COUNTS = (7, 6, 5)

KH_GRID = (
    0.1,
    0.3,
    np.pi / 6,
    0.8,
    1.0,
    np.pi / 3,  # T_bulk null (conventions Sec.5)
    1.5,
    np.pi / 2,  # T_full null (conventions Sec.5)
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
)

# mpmath cross-check subset: (kH, R_cut) pairs, kept small for runtime.
MPMATH_CHECK_POINTS = ((1.0, 40.0), (2.0, 80.0))
MPMATH_DPS_STAGES = (30, 50)

SCHEMA_VERSION = 1
OUTPUT_PATH = Path(__file__).parent / "data" / "halfspace_v1.json"

# Content-addressed provenance.
# Replaces the earlier design where ``generated_from_commit`` was verified
# via `git cat-file`/`git merge-base --is-ancestor` against the current
# repository -- structurally incompatible with a history-free clean-repo
# history-truncated checkout (see tests/_provenance_contract.py module docstring
# for the full rationale). ``upstream_revision`` below is now a declarative
# attribution value only, never verified against git history by the
# standard provenance checks.
ORACLE_NAME = "halfspace_v1"
ORACLE_METHOD_VERSION = "1.0"  # bump on derivation-*method* changes, not source edits
UPSTREAM_REPOSITORY = (
    "gragra (this repository) -- independent in-repo derivation from the "
    "published closed form (see docs/architecture/halfspace_conventions.md), "
    "not copied from an external upstream project"
)


def disk_az(d: float, r_cut: float, sigma: float) -> float:
    """On-axis Newtonian gravity (z-component) of a finite uniform disk.

    d = z_obs - z_disk (signed); r_cut = disk radius; sigma = areal density.
    See module docstring for derivation.
    """
    sign_d = 1.0 if d > 0.0 else (-1.0 if d < 0.0 else 0.0)
    return -2.0 * np.pi * G * sigma * (sign_d - d / np.sqrt(d * d + r_cut * r_cut))


def _bulk_integrand(z_prime: float, h: float, r_cut: float) -> float:
    d = -h - z_prime
    delta_rho = 2.0 * RHO_0 * XI_INC * K * np.sin(K * z_prime)
    return disk_az(d, r_cut, delta_rho)


def bulk_finite(
    h: float, r_cut: float, z_max_factor: float = Z_MAX_FACTOR
) -> tuple[float, float]:
    """Truncated bulk gravity integral (scipy float64 QUADPACK).

    ``points=[-h]`` is passed explicitly: the integrand is non-smooth at
    z'=-h (the disk's own plane, where the ``sign(d)`` term in
    :func:`disk_az` jumps), and QUADPACK's adaptive subdivision should be
    told about a known breakpoint rather than relying on it to locate one.
    """
    z_max = z_max_factor * r_cut
    val, err = integrate.quad(
        _bulk_integrand,
        -z_max,
        0.0,
        args=(h, r_cut),
        points=[-h],
        limit=QUAD_LIMIT,
    )
    return val, err


def z_max_sensitivity(h: float) -> dict:
    """Re-evaluate the bulk integral at :data:`Z_MAX_SENSITIVITY_R_CUT` for
    each factor in :data:`Z_MAX_SENSITIVITY_FACTORS`, so the Z_max choice
    used for the main R_cut sweep is reproducibly re-verified rather than
    only justified by an undocumented scratch script (see module
    docstring)."""
    r_cut = Z_MAX_SENSITIVITY_R_CUT
    values = [bulk_finite(h, r_cut, factor)[0] for factor in Z_MAX_SENSITIVITY_FACTORS]
    return {
        "r_cut": r_cut,
        "z_max_factor_stages": list(Z_MAX_SENSITIVITY_FACTORS),
        "bulk_values": values,
        "tail_error_estimate": abs(values[-1] - values[-2]),
    }


def sheet_finite(h: float, r_cut: float) -> float:
    """Truncated free-surface mass-sheet gravity contribution (exact algebra
    given r_cut; no quadrature error)."""
    sigma_surface = 2.0 * RHO_0 * XI_INC
    return disk_az(-h, r_cut, sigma_surface)


def _richardson_model(r_cut, c0, c1, c2):
    return c0 + c1 / r_cut + c2 / r_cut**2


def extrapolate(r_cuts: np.ndarray, values: np.ndarray) -> tuple[float, float, float]:
    """Fit c0 + c1/R + c2/R^2 by least squares; initial guess derived only
    from the largest-R_cut sample (never from any closed-form target)."""
    p0 = [values[-1], 0.0, 0.0]
    popt, _ = curve_fit(_richardson_model, r_cuts, values, p0=p0, maxfev=20000)
    return float(popt[0]), float(popt[1]), float(popt[2])


def fit_stability(r_cuts: np.ndarray, values: np.ndarray) -> dict:
    """Re-run :func:`extrapolate` on the last 7/6/5 R_cut stages (dropping
    the smallest, least-converged stages) and report the largest pairwise
    difference among the resulting c0 values, so extrapolation stability is
    reproducibly re-verified rather than assumed from a single 7-stage fit
    (see module docstring)."""
    c0_by_count = {}
    for count in FIT_STABILITY_STAGE_COUNTS:
        c0, _, _ = extrapolate(r_cuts[-count:], values[-count:])
        c0_by_count[str(count)] = c0
    c0s = list(c0_by_count.values())
    max_diff = max(abs(a - b) for a in c0s for b in c0s)
    return {
        "extrapolated_r_cut_inf_by_stage_count": c0_by_count,
        "fit_stability_max_diff": max_diff,
    }


def bulk_finite_mpmath(h: float, r_cut: float, dps: int) -> float:
    """mpmath cross-check of :func:`bulk_finite`, with explicit half-period
    breakpoints so ``mpmath.quad`` resolves the oscillatory integrand (it
    does not auto-subdivide a single interval the way scipy's adaptive
    QUADPACK does)."""
    mp.mp.dps = dps
    h_m, k_m, r_m = mp.mpf(h), mp.mpf(K), mp.mpf(r_cut)
    z_max = Z_MAX_FACTOR * r_m

    def integrand(z_prime):
        d = -h_m - z_prime
        delta_rho = 2 * RHO_0 * XI_INC * k_m * mp.sin(k_m * z_prime)
        sign_d = 1 if d > 0 else (-1 if d < 0 else 0)
        return -2 * mp.pi * G * delta_rho * (sign_d - d / mp.sqrt(d**2 + r_m**2))

    period = 2 * mp.pi / k_m
    half = period / 2

    left_pts = [-h_m]
    z = -h_m
    while z > -z_max:
        z -= half
        left_pts.insert(0, max(z, -z_max))
    if left_pts[0] != -z_max:
        left_pts.insert(0, -z_max)

    right_pts = []
    z = -h_m
    while z < 0:
        z += half
        right_pts.append(min(z, mp.mpf(0)))
    if not right_pts or right_pts[-1] != 0:
        right_pts.append(mp.mpf(0))

    points = left_pts + right_pts
    return float(mp.quad(integrand, points))


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
    """SHA-256 of this script's own source, recorded in the output JSON so a
    generator edit without regenerating the JSON is mechanically detectable
    (``tests/test_halfspace_gravity_oracle.py`` compares this against the
    saved value). Recorded as provenance key ``source_content_sha256``
    (see the module-level docstring)."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _derivation_input() -> dict:
    """Constants controlling this generator's numeric output, hashed
    separately from the whole-script ``generator_sha256``/
    ``source_content_sha256`` (which also changes on comment/docstring-only
    edits that don't affect the computed values). If a future edit moves
    these constants without changing their values, this hash intentionally
    stays the same -- it tracks *input values*, not *source layout*."""
    return {
        "G": G,
        "RHO_0": RHO_0,
        "XI_INC": XI_INC,
        "K": K,
        "KH_GRID": [float(v) for v in KH_GRID],
        "R_CUT_STAGES": list(R_CUT_STAGES),
        "Z_MAX_FACTOR": Z_MAX_FACTOR,
        "Z_MAX_SENSITIVITY_FACTORS": list(Z_MAX_SENSITIVITY_FACTORS),
        "Z_MAX_SENSITIVITY_R_CUT": Z_MAX_SENSITIVITY_R_CUT,
        "MPMATH_CHECK_POINTS": [list(p) for p in MPMATH_CHECK_POINTS],
        "MPMATH_DPS_STAGES": list(MPMATH_DPS_STAGES),
    }


def derivation_input_sha256() -> str:
    """SHA-256 of a canonical JSON encoding of ``_derivation_input()``."""
    encoded = json.dumps(_derivation_input(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def golden_artifact_sha256(non_provenance_body: dict) -> str:
    """SHA-256 of a canonical JSON encoding of the generated data excluding
    the ``provenance`` block itself (see
    ``tests._provenance_contract.golden_artifact_sha256``, which this
    duplicates rather than imports so this script stays runnable standalone
    -- ``python tests/oracles/generate_halfspace_oracle.py`` -- without
    ``tests/`` on ``sys.path``)."""
    encoded = json.dumps(non_provenance_body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def generate() -> dict:
    points_out = []
    for kh in KH_GRID:
        h = float(kh) / K
        r_cuts = np.array(R_CUT_STAGES, dtype=np.float64)

        bulk_vals = np.array([bulk_finite(h, r)[0] for r in r_cuts])
        bulk_errs = np.array([bulk_finite(h, r)[1] for r in r_cuts])
        sheet_vals = np.array([sheet_finite(h, r) for r in r_cuts])
        full_vals = bulk_vals + sheet_vals

        bulk_c0, bulk_c1, bulk_c2 = extrapolate(r_cuts, bulk_vals)
        sheet_c0, sheet_c1, sheet_c2 = extrapolate(r_cuts, sheet_vals)
        full_c0, full_c1, full_c2 = extrapolate(r_cuts, full_vals)

        points_out.append(
            {
                "kH": float(kh),
                "H": h,
                "r_cut_stages": r_cuts.tolist(),
                "z_max_factor": Z_MAX_FACTOR,
                "z_max_sensitivity": z_max_sensitivity(h),
                "bulk": {
                    "values_by_r_cut": bulk_vals.tolist(),
                    "quad_error_estimate_by_r_cut": bulk_errs.tolist(),
                    "extrapolated_r_cut_inf": bulk_c0,
                    "fit_c1": bulk_c1,
                    "fit_c2": bulk_c2,
                    **fit_stability(r_cuts, bulk_vals),
                },
                "sheet": {
                    "values_by_r_cut": sheet_vals.tolist(),
                    "extrapolated_r_cut_inf": sheet_c0,
                    "fit_c1": sheet_c1,
                    "fit_c2": sheet_c2,
                    **fit_stability(r_cuts, sheet_vals),
                },
                "full": {
                    "values_by_r_cut": full_vals.tolist(),
                    "extrapolated_r_cut_inf": full_c0,
                    "fit_c1": full_c1,
                    "fit_c2": full_c2,
                    **fit_stability(r_cuts, full_vals),
                },
            }
        )

    mpmath_cross_check = []
    for kh, r_cut in MPMATH_CHECK_POINTS:
        h = kh / K
        scipy_val, _ = bulk_finite(h, r_cut)
        mp_vals = {dps: bulk_finite_mpmath(h, r_cut, dps) for dps in MPMATH_DPS_STAGES}
        mpmath_cross_check.append(
            {
                "kH": kh,
                "r_cut": r_cut,
                "scipy_float64_value": scipy_val,
                "mpmath_values_by_dps": {str(d): v for d, v in mp_vals.items()},
            }
        )

    body = {
        "schema_version": SCHEMA_VERSION,
        "description": (
            "Independent Newton-gravity-integral oracle for "
            "gragra.analytic.halfspace.halfspace_transfer_bulk/full "
            "(half-space oracle). See the generator module docstring for "
            "the full derivation and extrapolation contract."
        ),
        "definition": {
            "relation": "delta_a_z(H) = -4*pi*G*rho_0*xi_inc*T(kH)",
            "dimensionless_units": {"G": G, "rho_0": RHO_0, "xi_inc": XI_INC, "k": K},
            "note": (
                "T = -(bulk+sheet raw value extrapolated) / "
                "(4*pi*G*rho_0*xi_inc); see the test file for the exact "
                "comparison to gragra's T_bulk/T_full."
            ),
        },
        "extrapolation_method": (
            "least-squares fit f(R)=c0+c1/R+c2/R^2 over r_cut_stages, "
            "initial guess from the largest-R_cut sample only (no "
            "closed-form target used as fit input); c0 is the reported "
            "R_cut -> infinity value. Stability re-verified by refitting on "
            "the last 6 and last 5 stages (see per-point "
            "extrapolated_r_cut_inf_by_stage_count / fit_stability_max_diff). "
            "Vertical (Z_max) truncation re-verified separately per point "
            "via z_max_sensitivity (see module docstring)."
        ),
        "points": points_out,
        "mpmath_cross_check": mpmath_cross_check,
    }

    provenance = {
        "oracle_name": ORACLE_NAME,
        "oracle_method_version": ORACLE_METHOD_VERSION,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_revision": _git_head_sha(),
        "source_content_sha256": generator_sha256(),
        "derivation_input_sha256": derivation_input_sha256(),
        "golden_artifact_sha256": golden_artifact_sha256(body),
        "generator_script": "tests/oracles/generate_halfspace_oracle.py",
        "independent_of": ["gragra.analytic", "gragra.waves"],
        "convention_doc": "docs/architecture/halfspace_conventions.md#5",
        "not_a_cross_validation_against_external_toolbox_execution": True,
    }

    return {
        "schema_version": body["schema_version"],
        "description": body["description"],
        "provenance": provenance,
        "definition": body["definition"],
        "extrapolation_method": body["extrapolation_method"],
        "points": body["points"],
        "mpmath_cross_check": body["mpmath_cross_check"],
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
