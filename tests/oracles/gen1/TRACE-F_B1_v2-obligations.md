# TRACE-F_B1_v2-obligations: 8 SPECIFIED_PENDING_EXECUTION obligations

Lane B (seismic implementation owner). HOLD partial release:
implementation / unit self-checks / review preparation ONLY.
E_B1 / R_B1 / P_science remain PROHIBITED — no formal reference generation,
no E_B1 construction, no sealing, no production comparison.

Frozen inputs (read-only `git show` from protocol commit `e3a20eb`; protocol
worktree NOT modified; budget/fixtures UNCHANGED and untouched):

- `B1-procedure-v2.json` sha256
  `6e4aa87ff5d50cfdf1856e320aecfb847d283d9dd4307abaa2e247f73a340cd5`
- `B1-freeze-v2.json` (FROZEN; B-HOLD released for implementation/unit-tests/
  review only; E_B1/R_B1/P_science prohibited)
- `CHANGELOG-F_B1-v1-to-v2-candidate.md` (8 items 04-11 normative-defined,
  verification pending — this file records the verification)
- `B1-budget.json` `a3e321dc…` / `B1-fixtures.json` `829e0e2c…` (MATCHED via
  read-only `git show`; no change)

Implementation: `tests/oracles/gen1/b1_analytic_enclosure.py` (pure
`Fraction`/`int`; no float, mpmath, Decimal, `gragra` import, literal pi).
Pinned implementation blob SHA (post-RC3 comment-only): `cc9213c50b2e305391cde35d961220f478f32372`.
Tests: `tests/test_gen1_b1_analytic_enclosure.py` (61 checks).
Pinned test blob SHA: `dfa1c2d57270b59059d2f466c0e2d04f26d0a43d`.
`file:line` below is auxiliary only (line numbers drift); the primary anchor
per row is obligation_id + implementation blob SHA + implementation
symbol/function + test blob SHA + full test function name(s) + result.

| obligation_id | implementation anchor (blob SHA + symbol; file:line aux only) | test anchor (blob SHA + full test name; file:line aux only) | result |
|---|---|---|---|
| B1-ch04 | impl blob `cc9213c50b2e305391cde35d961220f478f32372`; symbols `atan_enclosure`, `machin_tangent_identities`, `verify_machin_tangent_identities`, `machin_branch_bounds`, `verify_machin_branch`, `assemble_pi`, `multiply_enclosure`, `square_enclosure`, `angular_moment_table`, `component_total_convert`, `transfer_sign` (aux `b1_analytic_enclosure.py:123,148,160,169,189,204,286,303,662,678,693`) | test blob `dfa1c2d57270b59059d2f466c0e2d04f26d0a43d`; `test_alternating_interval_parity`, `test_wrong_parity_bound_rejected`, `test_malformed_series_parameters_rejected`, `test_fixed_n_widths_and_nesting`, `test_directed_assembly_and_wrong_order_rejected`, `test_negative_multiplier_ordering`, `test_tiny_nonzero_vs_zero`, `test_malformed_input_rejected`, `test_sign_phase_conventions_consistent`, `test_moment_and_component_total_normalization`, `test_g1_g2_confusion_guard`, `test_square_bound_and_broken_rejected`, `test_center_outside_claimed_enclosure_rejected`, `test_downward_rounded_u_rejected`, `test_machin_tangent_identities_exact`, `test_machin_branch_inequalities_exact` (aux `test_gen1_b1_analytic_enclosure.py:72,90,104,122,130,148,165,185,208,221,237,270,346,375,48,57`) | PASS — 61/61 in file; parity even `[S,S+r]`/odd `[S-r,S]` + nesting over N in {1,2,3,4,7,8,30,100,128}; primary width < 1e-180, fallback < 1e-138 (integer-compared); wrong-order/wrong-parity/broken variants rejected |
| B1-ch05 | impl blob `cc9213c50b2e305391cde35d961220f478f32372`; symbols `isqrt_exact`, `sqrt_bracket`, `verify_sqrt_bracket`, `frobenius_upper` (aux `b1_analytic_enclosure.py:365,385,431,441`) | test blob `dfa1c2d57270b59059d2f466c0e2d04f26d0a43d`; `test_sqrt_brackets_certified`, `test_sqrt_exact_zero_branch`, `test_sqrt_tightening_nests_and_type_pure`, `test_frobenius_upper_bound` (aux `test_gen1_b1_analytic_enclosure.py:314,323,330,298`) | PASS — `lo^2 <= z <= hi^2` integer-square certificates; width ≤ 2^-59; z=0 → `(0,0)` incl. `-0.0`; negative radicand raises; `U_pi = sqrt(3r^2)+U_ser` covers from above |
| B1-ch06 | impl blob `cc9213c50b2e305391cde35d961220f478f32372`; symbols `_quantize_ties_even`, `format_canonical`, `format_upward`, `serialize_enclosure`, `reparse`, `check_no_narrowing`, `public_component_sum`, `CENTRAL_DIGITS` (aux `b1_analytic_enclosure.py:486,526,553,577,609,621,648; :40`) | test blob `dfa1c2d57270b59059d2f466c0e2d04f26d0a43d`; `test_central_transport_80_digits_ties_even`, `test_round_trip_never_narrows`, `test_public_component_sum_exact`, `test_downward_rounded_u_rejected`, `test_upward_u_exact_and_zero`, `test_malformed_record_rejected`, `test_edited_exact_endpoints_rejected` (aux `test_gen1_b1_analytic_enclosure.py:407,424,437,375,394,191,357`) | PASS — 80-digit central (ties-even by integer quotient/remainder), outward U covering BOTH endpoints (halfwidth + central rounding error), negative-zero `0e+0`, tiny `1e-100` preserved; reparse ⊇ original proven; downward-U/edited-center/edited-endpoint/malformed records all rejected |
| B1-ch07 | impl blob `cc9213c50b2e305391cde35d961220f478f32372`; symbols `allocation_single_owner`, `allocation_nonzero_requires_uncertainty`, `whole_observable_zero_certificate`, `joint_diagonal_enclosure`, `single_entry_shrink` (aux `b1_analytic_enclosure.py:778,802,811,339,349`) | test blob `dfa1c2d57270b59059d2f466c0e2d04f26d0a43d`; `test_allocation_single_ownership`, `test_zero_certificate_whole_observable_only`, `test_joint_correlation_rejects_single_entry_shrink`, `test_tiny_nonzero_vs_zero` (aux `test_gen1_b1_analytic_enclosure.py:470,501,288,165`) | PASS — tail owns pi-enclosure; quadrature/formulation/roundoff pairwise distinct; serialization owns central rounding; nonzero-c ⇒ U_pi > 0; partial-zero and nonzero-U zero certs rejected; joint (not single-entry) enclosure |
| B1-ch08 | impl blob `cc9213c50b2e305391cde35d961220f478f32372`; symbols `si_recompute`, `g2_confusion_guard` (aux `b1_analytic_enclosure.py:723,747`) | test blob `dfa1c2d57270b59059d2f466c0e2d04f26d0a43d`; `test_g1_g2_confusion_guard`, `test_si_recompute_separate_residual` (aux `test_gen1_b1_analytic_enclosure.py:237,249`) | PASS — `S_a^SI = G_SI^2 · S_a^{G-free}` from exact decimal-string G (test input, never production/binary64); own SI residual widens outward; G≤0/unordered/non-Fraction raise; mistagged representations raise |
| B1-ch09 | impl blob `cc9213c50b2e305391cde35d961220f478f32372`; symbols `aux_partition_check`, `aux_production_deny`, `aux_certificate_binding` (aux `b1_analytic_enclosure.py:824,840,852`; pins aux `:44`) | test blob `dfa1c2d57270b59059d2f466c0e2d04f26d0a43d`; `test_aux_partition_70_keys`, `test_aux_production_denying_and_certificate_binding` (aux `test_gen1_b1_analytic_enclosure.py:514,527`) | PASS — 56 `r_aux:*` + 14 extra = 70 exact (duplicates/short/wrong-split rejected); production/comparator_kernel/reference_kernel/calibration/baseline fields denied; S-null whole-field cert barred from second-order zero-scale scope |
| B1-ch10 | impl blob `cc9213c50b2e305391cde35d961220f478f32372`; symbol `ql_diagnostic_compare` (aux `b1_analytic_enclosure.py:866`; role `DEVELOPMENT_DIAGNOSTIC_NOT_ORACLE`, status `superseded as formal procedure`, `gates_pass_fail False`) | test blob `dfa1c2d57270b59059d2f466c0e2d04f26d0a43d`; `test_ql_diagnostic_cannot_gate` (aux `test_gen1_b1_analytic_enclosure.py:546`) | PASS — agreeing AND deliberately-absurd QL values yield the identical formal verdict; prior QL records untouched (no rename); R1 azimuth gate untouched (no R1 code in this lane addition); exact-integral gate-text update remains a freeze action (pending, not performed here) |
| B1-ch11 | impl blob `cc9213c50b2e305391cde35d961220f478f32372`; symbols `frozen_pins`, `verify_decision`, `demonstration_enclosure`, `aux_partition_check`, `aux_production_deny`, `aux_certificate_binding` (aux `b1_analytic_enclosure.py:59,889,905,824,840,852`) | test blob `dfa1c2d57270b59059d2f466c0e2d04f26d0a43d`; `test_decision_checks`, `test_fallback_decision_120`, `test_demonstration_deterministic`, `test_proof_path_purity_forbidden_tokens`, `test_runtime_types_exact`, `test_import_denial_and_no_production_read` (aux `test_gen1_b1_analytic_enclosure.py:449,454,459,565,597,608`) | PASS (review-prep only) — dataflow source is the pinned frozen hashes (no production/derivation leak); common-helper mapping has no new helpers (stdlib `Fraction`/`re` only); R_aux contract interface ready, sealing NOT performed; the review itself remains pending with Root |

## Declarations

- No budget/fixture/procedure changes: verified `a3e321dc…`/`829e0e2c…` read-only; new files only, nothing modified.
- No production inputs: no `gragra` import (subprocess import-denial test), no production outputs opened, no `references/`/`comparisons/` content touched.
- No formal generation: no E/R/P artifact; `demonstration_enclosure` docstring canonical wording is `implementation self-check artifact, NOT a formal reference (no E/R/P)` (TRACE quotes the impl verbatim; no `SELF_CHECK_ONLY_NOT_A_REFERENCE` token exists in impl or frozen procedure — RC2 aligned TRACE to impl, no runtime symbol added).
- QL diagnostic-only: `gates_pass_fail is False` asserted; absurd-QL invariance asserted.
- New files: `tests/oracles/gen1/b1_analytic_enclosure.py`,
  `tests/test_gen1_b1_analytic_enclosure.py`, this TRACE file.

## B1-ch11 resolution (appended; history above NOT rewritten)

- Previous state: review pending with Root (see B1-ch11 row above).
- Resolved-by: `reviews/b1-implementation-delta/review.json`.
- Decision: `APPROVE_FOR_E_B1_CONSTRUCTION`.
- Scope: implementation review only; the generator stack still requires
  separate approval.
- Lane note: this worktree builds the formal generator adapter only
  (`tests/oracles/gen1/b1_formal_generator.py` +
  `tests/test_gen1_b1_formal_generator.py`). No formal attempt was
  launched, no E_B1/R_B1/P_science artifact constructed, sealed, or
  compared. E_B1 construction remains a future gated action outside this
  lane's build.

## Generator revision: nonzero fixtures wired (appended; history above NOT rewritten)

- Revision: `reviewed_nonzero_compute` in
  `tests/oracles/gen1/b1_formal_generator.py` routes the 18 frozen nonzero
  science fixtures (`rho2700_component` / `rho1350_component` /
  `rho2700_total`) through the reviewed enclosure primitives only
  (exact decimal parsing, primary pi enclosure, directed
  scaling/squaring, component/total conversion, single-ownership gate,
  G1/G2 guard, independent SI recomputation). The reviewed implementation
  file `b1_analytic_enclosure.py` is NOT modified (blob still
  `cc9213c50b2e305391cde35d961220f478f32372`); the reviewed enclosure
  tests are NOT modified (blob still
  `dfa1c2d57270b59059d2f466c0e2d04f26d0a43d`).
- New tests: `tests/test_gen1_b1_nonzero_generator.py` (17 checks).
  Required coverage: 12 zero fixtures resolved exactly as the prior
  revision; all 18 nonzero reach reviewed-enclosure candidates with the
  exact 30 frozen keys; unresolved retained with error records (never
  dropped); production-blind maintained; no reference-path import;
  serialization preservation proven for every resolved record;
  malformed/non-enclosed inputs fail closed. All-18-resolved is NOT a
  success criterion — honest unresolved stays unresolved.
- Build-only outcome observed (no formal run to confirm): 30/30 records
  resolved in the isolated build (`12 zero + 18 nonzero`), keys exact,
  every resolved record passes `check_no_narrowing`. Formal admission is
  NOT claimed and remains unexecuted.
- Declarations: procedure/budget/fixtures unchanged (pins re-verified by
  the adapter at build); enclosure arithmetic, uncertainty allocation,
  and serialization contract unchanged; zero-fixture behavior unchanged
  (byte-identical enclosures vs prior revision); b1-001 and every sealed
  artifact untouched; E_B1_v2 not used for any new formal attempt; no
  formal attempt launched; no production read.
