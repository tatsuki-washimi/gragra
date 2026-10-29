"""Independent Newton-gravity-integral oracle generator for the Rayleigh-wave
Newtonian-noise acceleration above a homogeneous elastic half-space.

This script independently re-derives (from elasticity + Newton's law, NOT by
importing ``gragra``) the same physical result as the published closed form

    delta_a_hat(x0, w) = 2*pi*G*rho*gamma*exp(-k_R*h)*(i*e_k - e_z),   (ANNA eq.75)
    gamma = k_R*(1 - sqrt(k_zp/k_zs)) / (i*k_zp - k_R*sqrt(k_zp/k_zs))  (eq.76)

(P. Reumers et al., "ANNA: a toolbox for Newtonian Noise Analysis",
arXiv:2603.15157v4 Sec.4.3, = Harms 2019). It imports neither
``gragra.analytic`` nor ``gragra.waves`` -- it re-solves the Rayleigh secular
equation and re-derives the eigenfunctions from scratch (using only the
shared physical convention in ``docs/architecture/halfspace_conventions.md``
Sec.3, using the same convention as the P-wave oracle).

Reference note: this compares against the *published closed-form formula*
(re-derived from first principles below), not against ANNA toolbox execution
output.

Existing gragra fixture and this oracle's relationship
---------------------------------------------------------
``tests/test_anna_halfspace_fixture_contract.py`` already contains a
hand-computed evaluation of eq.75-76 using gragra's own q/s parametrization
(``rayleigh_speed``) -- that is a *self-consistency* check (Phase 16-b Stage
A), not independent verification: the same closed-form formula is evaluated
once and compared to itself. This oracle is different in kind: it computes
the same physical quantity via a completely different route (a finite-
truncated Newton volume/surface integral of the wave-induced mass
perturbation, extrapolated to the infinite-domain limit) that shares no
algebra with the closed form at all, only converging to it in the limit.

Physical setup and coordinate convention
------------------------------------------
z-up, free surface at z=0, elastic half-space z<=0 (h=-z>=0), matching
``docs/architecture/halfspace_conventions.md`` Sec.1-3. A Rayleigh wave
propagates along +x with wavenumber k>0, phase speed c_R (root of the
secular equation R(c)=(2-c^2/v_S^2)^2-4*sqrt(1-c^2/v_P^2)*sqrt(1-c^2/v_S^2)=0
on 0<c<v_S), q=sqrt(1-c_R^2/v_P^2), s=sqrt(1-c_R^2/v_S^2). The traction-free
two-exponential eigenfunctions (same convention document, independently
re-coded here, not imported):

    u_x(x',z') =  i*k*A * [e^{-q k h} - (2 q s/(1+s^2)) e^{-s k h}] * e^{i k x'}
    u_z(x',z') =    q*k*A * [e^{-q k h} - (2/(1+s^2))     e^{-s k h}] * e^{i k x'}

with h=-z'>=0 and A chosen so u_z(x'=0,z'=0)=+1 (unit surface displacement,
matching both ANNA eq.73 and gragra's ``rayleigh_displacement`` default
``amplitude_m=1.0`` -- the physical setup is the same, just re-derived).

Observer: a test point at height H>0 *above* the free surface (vacuum side,
z=+H in this module's own H-is-height-above-ground convention -- distinct
from gragra's own z coordinate, which structurally rejects z>0 evaluation;
this oracle works entirely in its own independent coordinate frame and only
maps back to gragra's convention where the test file compares against
gragra's production APIs), directly above the horizontal phase origin
(x=0, y=0).

ANNA-to-gragra convention boundary
-----------------------------------
ANNA eq.75 is the published reference expression

    delta_a_hat_ANNA = 2*pi*G*rho*gamma*exp(-k_R*h)*(i*e_k - e_z).

The ANNA Rayleigh convention uses z-down coordinates and the implicit
``exp(+i*omega*t)`` time convention, so its spatial phasor is
``exp(-i*k*x)``. gragra uses z-up coordinates and
``exp(i*(k*x-omega*t))``. With ``J=diag(1,1,-1)``, positions map as
``x_gragra=J @ x_ANNA``. Rayleigh's unit surface displacement is normalized
positive in ANNA's z-down direction but positive in gragra's z-up direction;
therefore the phase-conjugated vector mapping includes an additional minus
sign: ``v_gragra=-conj(J @ v_ANNA)`` for displacement and acceleration.

This generator's finite-integral values and the existing JSON deliberately
retain their established ANNA-eq.75 component/phase reporting convention:
``a_z`` is the real coefficient of ``-e_z`` and ``a_x`` is the real
coefficient of ``i*e_k``. They are not production gragra vectors and must
not be sign-flipped here. The gragra production target is the separately
mapped vector above; the consuming fixture/production comparison applies
that distinction explicitly.

Mass-perturbation channels (bulk + free-surface sheet, kept separate)
--------------------------------------------------------------------
1. Bulk (mass conservation to first order, delta_rho = -rho_0 div(u)):
   substituting the eigenfunctions above and using div(curl) = 0 (the s-
   exponential/shear-potential part is divergence-free and drops out
   *exactly* -- this is an internal consistency check this script performs
   numerically before using the simplified form, see ``_verify_bulk_cancellation``):

       delta_rho(x',z') = D * e^{q k z'} * e^{i k x'},   z'<=0
       D = rho_0 * k^2 * A * (c_R/v_P)^2

2. Free-surface mass sheet (moving free surface exposes/covers density
   rho_0 over the surface's own vertical displacement, at z'=0):

       sigma(x') = rho_0 * u_z(x',0) = rho_0 * e^{i k x'}   (since u_z(0)=1)

Newton integral via analytic y'-reduction (the "radial reduction" analogue)
-----------------------------------------------------------------------------
Unlike the P-wave half-space case, the Rayleigh source is NOT
axially symmetric about the observer's vertical axis -- it is periodic in x'
and translation-invariant in y' (the wave propagates along +x only). The
disk-radius reduction used for the P-wave case does not apply; instead, this script
integrates the y' direction analytically in closed form (the translation-
invariant analogue of the radial reduction), using the standard result
``integral_{-inf}^{inf} dy / (a^2+y^2)^{3/2} = 2/a^2`` (a>0). For a source
element at (x',y',z') and observer at (0,0,H), this collapses the on-axis
(x=y=0) acceleration contribution of an *infinite-y' strip* at fixed (x',z')
to a closed 2D "line mass" kernel:

    d(a_z)/(dx' dz') = -2 G delta_rho(x',z') (H-z') / (x'^2 + (H-z')^2)
    d(a_x)/(dx' dz') =  2 G delta_rho(x',z') x'      / (x'^2 + (H-z')^2)

(and the analogous 1D kernel for the surface sheet, at z'=0). This is the
2D/"line source" Newtonian-gravity analogue of the finite-disk formula.
A symbolic (SymPy, on the secular-equation constraint surface
``(1+s^2)^2=4*q*s``) check that its infinite-x' limit reproduces eq.75-76
exactly was run in a throwaway scratch script during design (confirmed
``sp.simplify(mine - target) == 0``) and is NOT preserved as a repository
artifact or re-run by this script -- adding a symbolic-algebra dependency
(SymPy) solely to re-check one already-converged identity was judged
disproportionate (Limitation Over Perfection); the simpler, purely
algebraic eq.76-to-gamma substitution below (:func:`rayleigh_gamma`) *is*
kept as a runtime-checked artifact (:func:`_verify_gamma_mapping`), matching
:func:`_verify_bulk_cancellation`'s pattern. What this script actually
re-verifies numerically, every run, is that a genuinely *finite-truncated*
version of the y'-reduced integral, extrapolated x'_cut -> infinity,
converges to that same closed form -- this is the actual per-run evidence
the test suite depends on, independent of the one-time design-stage check.

Two truncations, extrapolated/checked separately
-------------------------------------------------
- **Horizontal extent x'_cut**: because delta_rho and sigma carry an
  oscillatory e^{i k x'} factor, a truncation at an arbitrary (non-period-
  aligned) x'_cut leaves a spectral-leakage residual that does NOT decay
  monotonically with x'_cut (verified empirically during design: truncating
  at arbitrary cutoffs gives an error sequence that wobbles by a factor of
  ~1.5-3.5 between successive doublings instead of shrinking cleanly).
  Truncating instead at exact multiples of the wave period
  (x'_cut = N * 2*pi/k) removes this leakage and gives a clean, smoothly
  decaying truncation-error sequence (verified: ratios of ~8x per doubling
  for the z-component, ~2x for the x-component, matching the O(1/x'^3) and
  O(1/x'^1) leading tail-decay rates of the two components' kernels). This
  script therefore always truncates at period-aligned x'_cut = N_stage *
  (2*pi/k), analogous in spirit to the P-wave oracle's explicit-breakpoint discipline
  for its own oscillatory integrand.
- **Vertical bulk extent z'_cut**: the bulk density decays as e^{q k z'}
  (purely exponential, no oscillation -- unlike the P-wave case, which had a
  non-decaying oscillatory bulk density requiring careful cancellation).
  z'_cut = Z_CUT_DECAY_LENGTHS / (q*k) is comfortably in the exponential
  floor (e^{-60} ~ 1e-27); :func:`z_cut_sensitivity` re-verifies this
  reproducibly per point (factor in Z_CUT_SENSITIVITY_FACTORS = {30,60,120}),
  mirroring the P-wave oracle's Z_max-sensitivity discipline.

Quadrature method: the x'-integrals use ``scipy.integrate.quad`` with
``weight='cos'``/``weight='sin'`` (Filon-type oscillatory quadrature) rather
than a plain adaptive quadrature over the oscillatory integrand directly --
verified during design to be both far faster and far more accurate for this
oscillatory kernel (a plain ``dblquad`` over the raw complex-exponential
integrand needed >200 QUADPACK subdivisions and still left a ~2.6e-3
relative residual at x'_cut~200, vs. <1e-12 with the weighted quadrature at
the same cutoff).

Extrapolation contract (fixed BEFORE looking at any closed-form target)
-------------------------------------------------------------------------
For bulk, sheet, and full (=bulk+sheet) separately, per (v_P/v_S, kH) point
and per component (a_z, a_x): evaluate at all x'_cut stages (7 period
multiples: 4,8,16,32,64,128,256), fit f(x'_cut)=c0+c1/x'_cut+c2/x'_cut^2 by
least squares, using ``curve_fit`` with an initial guess derived ONLY from
the largest-x'_cut computed value (c0_guess=f(x'_max), c1_guess=c2_guess=0)
-- the closed-form target value is never used as a fit input, only
afterward as the pass/fail comparison in
``tests/test_rayleigh_gravity_oracle.py``. Fit-stability (re-fitting on the
last 6/5 stages) is recorded the same way as the P-wave oracle.

mpmath precision cross-check
------------------------------
For one representative (v_P/v_S, kH) point (not the full grid -- mpmath's
oscillatory 2D quadrature at dps=30 takes ~30s for a single bulk component
at a single x'_cut stage, vs. ~0.02s for the equivalent scipy Filon
evaluation), the same bulk and surface integrals are re-evaluated with
``mpmath.quad`` at two working precisions (dps=30, dps=50), with explicit
half-period breakpoints (mirroring the P-wave oracle's mpmath breakpoint discipline).
Agreement with the scipy float64 value confirms the scipy result is not an
artifact of floating-point rounding.

Output
------
Writes ``tests/oracles/data/rayleigh_v1.json``. Provenance
(``source_content_sha256``, ``derivation_input_sha256``,
``golden_artifact_sha256``, and the declarative-only ``upstream_revision``)
and the JSON-committed-to-git rationale follow the same pattern established
by the P-wave oracle (``generate_halfspace_oracle.py`` module docstring) -- not
re-explained here to avoid drift between the two files' copies of the same
rationale.
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
K = 1.0  # fixed wavenumber; H = kH / K = kH in these units

VP_VS_RATIOS = (np.sqrt(3.0), 2.0)
KH_GRID = (0.1, 0.5, 1.0, 2.0, 4.0, 6.0)

PERIOD = 2.0 * np.pi / K
X_CUT_PERIOD_STAGES = (4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0)
QUAD_LIMIT = 200

Z_CUT_DECAY_LENGTHS = 60.0  # z'_cut = Z_CUT_DECAY_LENGTHS / (q*K)
Z_CUT_SENSITIVITY_FACTORS = (30.0, 60.0, 120.0)

FIT_STABILITY_STAGE_COUNTS = (7, 6, 5)

# mpmath cross-check: one representative point, kept small for runtime (see
# module docstring -- mpmath's oscillatory 2D quadrature is ~1500x slower
# than the scipy Filon route for this integrand).
MPMATH_CHECK_POINT = {"vp_vs_ratio": float(np.sqrt(3.0)), "kH": 1.0}
MPMATH_CHECK_X_CUT_STAGE = 8.0  # period multiples
MPMATH_DPS_STAGES = (30, 50)

SCHEMA_VERSION = 1
OUTPUT_PATH = Path(__file__).parent / "data" / "rayleigh_v1.json"

# Content-addressed provenance (see
# generate_halfspace_oracle.py's constants and _derivation_input() for the
# rationale, not re-explained here to avoid drift between the two files'
# copies of the same rationale).
ORACLE_NAME = "rayleigh_v1"
ORACLE_METHOD_VERSION = "1.0"  # bump on derivation-*method* changes, not source edits
UPSTREAM_REPOSITORY = (
    "gragra (this repository) -- independent in-repo derivation from the "
    "published closed form (see closed_form_reference below), not copied "
    "from an external upstream project"
)


def secular_residual(c_over_vs: float, vp_vs_ratio: float) -> float:
    """Rayleigh secular function R(c/v_S), independently coded (Aki & Richards
    standard form, no extra c^2/v_S^2 factor on the second term)."""
    s2 = c_over_vs * c_over_vs
    p2 = s2 / (vp_vs_ratio * vp_vs_ratio)
    return (2.0 - s2) ** 2 - 4.0 * np.sqrt(1.0 - p2) * np.sqrt(1.0 - s2)


def solve_rayleigh(
    vp_vs_ratio: float, tol: float = 1e-14
) -> tuple[float, float, float]:
    """Bisect for c_R/v_S on (0.5, 1-1e-9), return (c_r_over_vs, q, s)."""
    lo, hi = 0.5, 1.0 - 1e-9
    f_lo = secular_residual(lo, vp_vs_ratio)
    f_hi = secular_residual(hi, vp_vs_ratio)
    if f_lo * f_hi > 0.0:
        raise ValueError(
            f"solve_rayleigh: no sign change on (0.5,1) for vp_vs_ratio={vp_vs_ratio}"
        )
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        f_mid = secular_residual(mid, vp_vs_ratio)
        if f_lo * f_mid <= 0.0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    c_over_vs = 0.5 * (lo + hi)
    p2 = c_over_vs**2 / (vp_vs_ratio * vp_vs_ratio)
    q = float(np.sqrt(1.0 - p2))
    s = float(np.sqrt(1.0 - c_over_vs**2))
    return c_over_vs, q, s


def rayleigh_gamma(q: float, s: float) -> float:
    """Eq.76: gamma = k_R*(1-sqrt(k_zp/k_zs)) / (i*k_zp-k_R*sqrt(k_zp/k_zs)),
    simplified under the substitution k_zp=-i*k_R*q, k_zs=-i*k_R*s (standard
    evanescent-wave vertical wavenumbers): k_zp/k_zs=q/s (k_R and the -i
    factors cancel in the ratio) and i*k_zp=k_R*q, giving the k_R-free form
    gamma = (1-sqrt(q/s)) / (q-sqrt(q/s)) used here. This is a pure algebraic
    substitution (holds for any q,s, not only on the secular-equation
    surface) -- kept as a runtime-checked artifact by
    :func:`_verify_gamma_mapping` rather than asserted in prose, per
    ``scientific-workflow.md``'s Symbolic Derivation Verification rule."""
    root = np.sqrt(q / s)
    return (1.0 - root) / (q - root)


def _verify_gamma_mapping(q: float, s: float, atol: float = 1e-12) -> None:
    """Cross-check :func:`rayleigh_gamma`'s simplified k_R-free form against
    the *un-simplified* eq.76 (complex k_zp,k_zs, arbitrary k_R) evaluated
    independently -- this is not circular (two independently written
    expressions are compared), unlike re-deriving the same simplified
    formula and comparing it to itself."""
    k_r = 1.7  # arbitrary nonzero value; gamma must be independent of it
    k_zp = -1j * k_r * q
    k_zs = -1j * k_r * s
    root = np.sqrt(k_zp / k_zs)
    gamma_eq76 = k_r * (1.0 - root) / (1j * k_zp - k_r * root)
    gamma_simplified = rayleigh_gamma(q, s)
    residual = abs(gamma_eq76 - gamma_simplified)
    if residual > atol:
        raise AssertionError(
            f"gamma mapping check failed: eq76={gamma_eq76!r} vs "
            f"simplified={gamma_simplified!r}, residual={residual!r} "
            f"(q={q}, s={s})"
        )


def _eigen_constants(q: float, s: float, k: float) -> tuple[float, float]:
    """Return (A, D): A normalizes u_z(surface, x'=0)=+1; D is the bulk
    density-perturbation amplitude coefficient (see module docstring)."""
    f_z0 = (s * s - 1.0) / (1.0 + s * s)
    a_amp = 1.0 / (q * k * f_z0)
    d_amp = RHO_0 * k * k * a_amp * (1.0 - q * q)  # (c_R/v_P)^2 = 1 - q^2
    return a_amp, d_amp


def _verify_bulk_cancellation(q: float, s: float, atol: float = 1e-12) -> None:
    """Internal consistency check (not a gragra comparison): the shear-
    potential (s-exponential) contribution to div(u) must cancel exactly,
    since curl fields are divergence-free. If it does not, the simplified
    bulk density formula used below (D*e^{qkz'}, no s-dependence) is wrong.
    """
    coef_x = 2.0 * q * s / (1.0 + s * s)
    coef_z = 2.0 / (1.0 + s * s)
    residual = q * coef_z * s - coef_x
    if abs(residual) > atol:
        raise AssertionError(
            f"bulk shear-cancellation check failed: residual={residual!r} "
            f"(q={q}, s={s}) -- the div(curl)=0 identity should make this "
            "exactly zero"
        )


def bulk_finite(
    q: float, d_amp: float, k: float, h: float, x_cut: float, z_cut: float
) -> tuple[float, float]:
    """Truncated bulk Newton integral (a_z, a_x), scipy float64 Filon
    quadrature (see module docstring for the kernel derivation).

    Note: the bulk density depends only on q (not s) -- the shear-potential
    (s-exponential) part cancels exactly, see
    :func:`_verify_bulk_cancellation`."""

    def inner_cos(zp: float) -> float:
        a = h - zp
        val, _ = integrate.quad(
            lambda xp: a / (xp * xp + a * a),
            -x_cut,
            x_cut,
            weight="cos",
            wvar=k,
            limit=QUAD_LIMIT,
        )
        return val

    def inner_sin(zp: float) -> float:
        a = h - zp
        val, _ = integrate.quad(
            lambda xp: xp / (xp * xp + a * a),
            -x_cut,
            x_cut,
            weight="sin",
            wvar=k,
            limit=QUAD_LIMIT,
        )
        return val

    az, _ = integrate.quad(
        lambda zp: d_amp * np.exp(q * k * zp) * (-2.0 * G) * inner_cos(zp),
        -z_cut,
        0.0,
        limit=QUAD_LIMIT,
    )
    ax, _ = integrate.quad(
        lambda zp: d_amp * np.exp(q * k * zp) * (2.0 * G) * inner_sin(zp),
        -z_cut,
        0.0,
        limit=QUAD_LIMIT,
    )
    return az, ax


def surface_finite(k: float, h: float, x_cut: float) -> tuple[float, float]:
    """Truncated free-surface mass-sheet Newton integral (a_z, a_x)."""
    az, _ = integrate.quad(
        lambda xp: -2.0 * G * RHO_0 * h / (xp * xp + h * h),
        -x_cut,
        x_cut,
        weight="cos",
        wvar=k,
        limit=QUAD_LIMIT,
    )
    ax, _ = integrate.quad(
        lambda xp: 2.0 * G * RHO_0 * xp / (xp * xp + h * h),
        -x_cut,
        x_cut,
        weight="sin",
        wvar=k,
        limit=QUAD_LIMIT,
    )
    return az, ax


def _richardson_model(x_cut, c0, c1, c2):
    return c0 + c1 / x_cut + c2 / x_cut**2


def extrapolate(x_cuts: np.ndarray, values: np.ndarray) -> tuple[float, float, float]:
    """Fit c0 + c1/x + c2/x^2 by least squares; initial guess derived only
    from the largest-x'_cut sample (never from any closed-form target)."""
    p0 = [values[-1], 0.0, 0.0]
    popt, _ = curve_fit(_richardson_model, x_cuts, values, p0=p0, maxfev=20000)
    return float(popt[0]), float(popt[1]), float(popt[2])


def fit_stability(x_cuts: np.ndarray, values: np.ndarray) -> dict:
    """Re-run :func:`extrapolate` on the last 7/6/5 x'_cut stages, recording
    the largest pairwise difference among the resulting c0 values (mirrors
    the P-wave oracle's fit_stability)."""
    c0_by_count = {}
    for count in FIT_STABILITY_STAGE_COUNTS:
        c0, _, _ = extrapolate(x_cuts[-count:], values[-count:])
        c0_by_count[str(count)] = c0
    c0s = list(c0_by_count.values())
    max_diff = max(abs(a - b) for a in c0s for b in c0s)
    return {
        "extrapolated_x_cut_inf_by_stage_count": c0_by_count,
        "fit_stability_max_diff": max_diff,
    }


def z_cut_sensitivity(q: float, d_amp: float, k: float, h: float) -> dict:
    """Re-evaluate the bulk integral at a fixed (moderate) x'_cut stage for
    each Z_CUT_DECAY_LENGTHS factor, recording the gap between the two
    largest factors as component-wise tail-truncation error estimates
    (mirrors the P-wave oracle's z_max_sensitivity). ``tail_error_estimate`` remains a
    backward-compatible alias for ``az_tail_error_estimate``."""
    x_cut = X_CUT_PERIOD_STAGES[2] * PERIOD  # moderate stage (16 periods)
    az_values = []
    ax_values = []
    for factor in Z_CUT_SENSITIVITY_FACTORS:
        z_cut = factor / (q * k)
        az, ax = bulk_finite(q, d_amp, k, h, x_cut, z_cut)
        az_values.append(az)
        ax_values.append(ax)
    az_tail_error = abs(az_values[-1] - az_values[-2])
    ax_tail_error = abs(ax_values[-1] - ax_values[-2])
    return {
        "x_cut": x_cut,
        "z_cut_decay_length_factors": list(Z_CUT_SENSITIVITY_FACTORS),
        "bulk_az_values": az_values,
        "bulk_ax_values": ax_values,
        "az_tail_error_estimate": az_tail_error,
        "ax_tail_error_estimate": ax_tail_error,
        # Preserve the v1 key for consumers that only checked the z tail.
        "tail_error_estimate": az_tail_error,
    }


def _mpmath_bulk_surface(
    q: float, d_amp: float, k: float, h: float, x_cut: float, dps: int
) -> dict:
    """mpmath cross-check of :func:`bulk_finite` / :func:`surface_finite` at
    one (x_cut, dps) stage, with explicit half-period breakpoints (mpmath's
    quad does not auto-subdivide to resolve oscillation the way scipy's
    adaptive QUADPACK / Filon routes do)."""
    mp.mp.dps = dps
    q_m, k_m, h_m = mp.mpf(q), mp.mpf(k), mp.mpf(h)
    d_m = mp.mpf(d_amp)
    x_cut_m = mp.mpf(x_cut)
    z_cut_m = mp.mpf(Z_CUT_DECAY_LENGTHS) / (q_m * k_m)

    period = 2 * mp.pi / k_m
    half = period / 2
    x_breaks = [mp.mpf(0)]
    z = mp.mpf(0)
    while z < x_cut_m:
        z += half
        x_breaks.append(min(z, x_cut_m))
    x_breaks_full = [-p for p in reversed(x_breaks)][:-1] + x_breaks

    def bulk_az_integrand(zp):
        a = h_m - zp

        def f(xp):
            return mp.cos(k_m * xp) * a / (xp**2 + a**2)

        return d_m * mp.e ** (q_m * k_m * zp) * (-2 * G) * mp.quad(f, x_breaks_full)

    def bulk_ax_integrand(zp):
        a = h_m - zp

        def f(xp):
            return mp.sin(k_m * xp) * xp / (xp**2 + a**2)

        return d_m * mp.e ** (q_m * k_m * zp) * (2 * G) * mp.quad(f, x_breaks_full)

    z_breaks = [-z_cut_m, -z_cut_m / 4, mp.mpf(0)]
    bulk_az = float(mp.quad(bulk_az_integrand, z_breaks))
    bulk_ax = float(mp.quad(bulk_ax_integrand, z_breaks))

    def surf_az_f(xp):
        return -2 * G * RHO_0 * h_m / (xp**2 + h_m**2)

    def surf_ax_f(xp):
        return 2 * G * RHO_0 * xp / (xp**2 + h_m**2)

    surf_az = float(mp.quad(lambda xp: mp.cos(k_m * xp) * surf_az_f(xp), x_breaks_full))
    surf_ax = float(mp.quad(lambda xp: mp.sin(k_m * xp) * surf_ax_f(xp), x_breaks_full))

    return {
        "bulk_az": bulk_az,
        "bulk_ax": bulk_ax,
        "surf_az": surf_az,
        "surf_ax": surf_ax,
    }


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
    """SHA-256 of this script's own source (see the P-wave oracle's
    ``generate_halfspace_oracle.generator_sha256`` for the rationale, not
    re-explained here). Recorded as provenance key ``source_content_sha256``
    (see the module-level docstring)."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _derivation_input() -> dict:
    """Constants controlling this generator's numeric output (see
    ``generate_halfspace_oracle._derivation_input`` for the rationale, not
    re-explained here). Distinct from ``generator_sha256``/
    ``source_content_sha256``, which hashes the whole script including
    comments/docstrings that don't affect the computed values."""
    return {
        "G": G,
        "RHO_0": RHO_0,
        "K": K,
        "VP_VS_RATIOS": [float(v) for v in VP_VS_RATIOS],
        "KH_GRID": list(KH_GRID),
        "X_CUT_PERIOD_STAGES": list(X_CUT_PERIOD_STAGES),
        "Z_CUT_DECAY_LENGTHS": Z_CUT_DECAY_LENGTHS,
        "Z_CUT_SENSITIVITY_FACTORS": list(Z_CUT_SENSITIVITY_FACTORS),
        "FIT_STABILITY_STAGE_COUNTS": list(FIT_STABILITY_STAGE_COUNTS),
        "MPMATH_CHECK_POINT": MPMATH_CHECK_POINT,
        "MPMATH_CHECK_X_CUT_STAGE": MPMATH_CHECK_X_CUT_STAGE,
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
    -- ``python tests/oracles/generate_rayleigh_oracle.py`` -- without
    ``tests/`` on ``sys.path``)."""
    encoded = json.dumps(non_provenance_body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def generate() -> dict:
    points_out = []
    for vp_vs_ratio in VP_VS_RATIOS:
        _, q, s = solve_rayleigh(vp_vs_ratio)
        _verify_bulk_cancellation(q, s)
        _verify_gamma_mapping(q, s)
        gamma = rayleigh_gamma(q, s)
        _, d_amp = _eigen_constants(q, s, K)

        for kh in KH_GRID:
            h = float(kh) / K
            x_cuts = np.array(
                [n * PERIOD for n in X_CUT_PERIOD_STAGES], dtype=np.float64
            )
            z_cut = Z_CUT_DECAY_LENGTHS / (q * K)

            bulk_az_vals = np.empty_like(x_cuts)
            bulk_ax_vals = np.empty_like(x_cuts)
            surf_az_vals = np.empty_like(x_cuts)
            surf_ax_vals = np.empty_like(x_cuts)
            for i, x_cut in enumerate(x_cuts):
                bulk_az_vals[i], bulk_ax_vals[i] = bulk_finite(
                    q, d_amp, K, h, x_cut, z_cut
                )
                surf_az_vals[i], surf_ax_vals[i] = surface_finite(K, h, x_cut)
            full_az_vals = bulk_az_vals + surf_az_vals
            full_ax_vals = bulk_ax_vals + surf_ax_vals

            channels = {}
            for name, az_vals, ax_vals in (
                ("bulk", bulk_az_vals, bulk_ax_vals),
                ("sheet", surf_az_vals, surf_ax_vals),
                ("full", full_az_vals, full_ax_vals),
            ):
                az_c0, az_c1, az_c2 = extrapolate(x_cuts, az_vals)
                ax_c0, ax_c1, ax_c2 = extrapolate(x_cuts, ax_vals)
                channels[name] = {
                    "az_values_by_x_cut": az_vals.tolist(),
                    "ax_values_by_x_cut": ax_vals.tolist(),
                    "az_extrapolated_x_cut_inf": az_c0,
                    "az_fit_c1": az_c1,
                    "az_fit_c2": az_c2,
                    "ax_extrapolated_x_cut_inf": ax_c0,
                    "ax_fit_c1": ax_c1,
                    "ax_fit_c2": ax_c2,
                    "az_fit_stability": fit_stability(x_cuts, az_vals),
                    "ax_fit_stability": fit_stability(x_cuts, ax_vals),
                }

            points_out.append(
                {
                    "vp_vs_ratio": float(vp_vs_ratio),
                    "kH": float(kh),
                    "H": h,
                    "q": q,
                    "s": s,
                    "gamma": gamma,
                    "x_cut_period_stages": list(X_CUT_PERIOD_STAGES),
                    "z_cut": z_cut,
                    "z_cut_sensitivity": z_cut_sensitivity(q, d_amp, K, h),
                    **channels,
                }
            )

    mp_point = MPMATH_CHECK_POINT
    _, q_mp, s_mp = solve_rayleigh(mp_point["vp_vs_ratio"])
    _, d_amp_mp = _eigen_constants(q_mp, s_mp, K)
    h_mp = mp_point["kH"] / K
    x_cut_mp = MPMATH_CHECK_X_CUT_STAGE * PERIOD
    scipy_bulk_az, scipy_bulk_ax = bulk_finite(
        q_mp, d_amp_mp, K, h_mp, x_cut_mp, Z_CUT_DECAY_LENGTHS / (q_mp * K)
    )
    scipy_surf_az, scipy_surf_ax = surface_finite(K, h_mp, x_cut_mp)
    mpmath_by_dps = {
        str(dps): _mpmath_bulk_surface(q_mp, d_amp_mp, K, h_mp, x_cut_mp, dps)
        for dps in MPMATH_DPS_STAGES
    }
    mpmath_cross_check = {
        "vp_vs_ratio": mp_point["vp_vs_ratio"],
        "kH": mp_point["kH"],
        "x_cut": x_cut_mp,
        "scipy_float64": {
            "bulk_az": scipy_bulk_az,
            "bulk_ax": scipy_bulk_ax,
            "surf_az": scipy_surf_az,
            "surf_ax": scipy_surf_ax,
        },
        "mpmath_values_by_dps": mpmath_by_dps,
    }

    body = {
        "schema_version": SCHEMA_VERSION,
        "description": (
            "Independent Newton-gravity-integral oracle for the Rayleigh-wave "
            "half-space Newtonian-noise acceleration, compared "
            "against ANNA eqs.75-76 (arXiv:2603.15157v4 = Harms 2019). See "
            "generate_rayleigh_oracle.py module docstring for the full "
            "derivation and extrapolation contract."
        ),
        "definition": {
            "relation": (
                "a(H) = 2*pi*G*rho_0*gamma*exp(-kH)*(i*e_x - e_z), "
                "with e_x the propagation direction and e_z vertical "
                "(gamma = eq.76). This oracle reports a_z (real) and the "
                "imaginary coefficient of a_x (i.e. a_x/i, real) separately."
            ),
            "anna_to_gragra_mapping": {
                "anna_phase_and_axes": (
                    "ANNA eq.75 uses z-down coordinates and exp(-i*k*x) "
                    "with implicit exp(+i*omega*t)."
                ),
                "gragra_phase_and_axes": (
                    "gragra uses z-up coordinates and exp(i*(k*x-omega*t))."
                ),
                "position_mapping": "x_gragra = J @ x_ANNA, J=diag(1,1,-1)",
                "rayleigh_vector_mapping": "v_gragra = -conj(J @ v_ANNA)",
                "normalization_sign": (
                    "The leading minus changes ANNA's positive z-down surface "
                    "normalization to gragra's positive z-up normalization."
                ),
                "oracle_reporting": (
                    "Saved oracle values preserve the existing ANNA-eq.75 "
                    "component/phase reporting and are not sign-flipped into "
                    "gragra production vectors."
                ),
                "gragra_target": (
                    "Production comparisons target the separately mapped "
                    "gragra vector v_gragra, not a reinterpretation of the "
                    "stored oracle scalars."
                ),
            },
            "z_cut_sensitivity": (
                "bulk_az_values and bulk_ax_values store the sensitivity sweep; "
                "az_tail_error_estimate and ax_tail_error_estimate are the "
                "absolute differences of their final two values. "
                "tail_error_estimate is a backward-compatible alias for "
                "az_tail_error_estimate."
            ),
            "dimensionless_units": {"G": G, "rho_0": RHO_0, "k": K},
        },
        "extrapolation_method": (
            "least-squares fit f(x)=c0+c1/x+c2/x^2 over period-aligned "
            "x_cut_period_stages, initial guess from the largest-x_cut "
            "sample only (no closed-form target used as fit input); c0 is "
            "the reported x_cut -> infinity value. Stability re-verified by "
            "refitting on the last 6 and last 5 stages. Vertical (z_cut) "
            "truncation re-verified separately via z_cut_sensitivity."
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
        "generator_script": "tests/oracles/generate_rayleigh_oracle.py",
        "independent_of": ["gragra.analytic", "gragra.waves"],
        "convention_doc": "docs/architecture/halfspace_conventions.md#3",
        "closed_form_reference": "arXiv:2603.15157v4 eqs.75-76 (Sec.4.3)",
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
