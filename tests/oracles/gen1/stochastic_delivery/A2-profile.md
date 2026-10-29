# A2 retained evidence profile

This page documents two distinct selected-delivery paths. The existing
`a2-ledger-v1` below applies only when an actual managed production comparison
was run. The additive selected-delivery v2 closure path is documented at the end;
it represents the Option A resource-limited decision without a comparison.

`a2-ledger-v1` is a separate offline profile. It never accepts a generic A2
descriptor or re-executes production, the oracle, or the operational adapter.
It retains the original E60 reference authorization and its ten projected pins;
the formatted runtime budget is separately pinned in the actual comparison
closure. The frozen partition is 84 records: 54 science and 30 auxiliary,
including 56 physical CSD records and 28 exact-nominal scalar records.

The operational version selector recognizes exact budget/adapter byte pairs:
the retained formatted runtime `f46f41`/`bd7af6` pair and the stricter-cap
successor `a61ed6`/`1cf51b` pair. Adopting the successor still requires its actual
three source reviews and the exact comparison authority; this reader source
does not issue that admission. Other hashes and mixed pairs fail. The original
reference budget `28719b` remains an immutable reference-provenance identity.

A bounded descriptor has the existing `profile` and one-element `stages` list.
Its stage has the usual twelve fields plus `managed_controller_completion`.
The completion is the actual root-issued eleven-field resource packet. Clock
snapshots use a separate finite operational JSON decoder; scientific JSON keeps
its existing prohibition on floating literals. The local validator proof must
name the actual admitted `family_supervisor.verify_family_result`, with its
`managed_controller_completion_pin` argument, and must not claim reexecution
from CAS relocation.

Worker elapsed time must remain within 3600 seconds. The declared controller
deadline is an integer from 1 through 3600, its wall bound is that deadline plus
100 seconds and never exceeds 3700, and its actual span cannot exceed the
declared wall bound. A stricter declared operation cap, such as 300/400, remains
valid; it cannot enlarge either frozen maximum.

Partial evidence adds `partial_eligibility` to the descriptor. Its stage's
`result_admission` is that same eligibility pin and `local_validator_proof` is
null: the PASS-only execution validator has not established bounded acceptance.
The eligibility packet has schema `gen1-root-a2-partial-eligibility-v1` and exact
fields defined by `verify_partial_evidence`. It records the actual attempt,
supervisor, run pins, four frozen-artifact hashes, raw result hash, exact status
counts, ordinary three result-review pins and actual author exclusions. It also
pins the retained frozen-budget exhaustion evidence. Root may issue it only
after those existing independent reviewers establish the previously permitted
numerical-conditioning partial outcome. This introduces no additional review
roles or numerical acceptance limits.

The parser preserves failed and unresolved keys. Its summary says
`PARTIAL_NUMERICAL_CONDITIONING_UNRESOLVED`, keeps numerical delivery and bounded
result admission false, and identifies independent partial-evidence review
separately. Execution failure alone does not establish eligibility. No actual
partial packet is manufactured by the reader or asserted by this specification.

`passed_counts` counts only independently replayed frozen assessments. Separate
`retained_claimed_passed_counts` and `retained_claimed_failed_count` describe the
serialized row flags, and `unreplayed_resolved_count` identifies quantities whose
Gfree/SI counterpart prevents replay. Such quantities never contribute to
independent passes. Eligibility `counts` binds the retained row dispositions;
that binding does not upgrade them to numerical acceptance.

Both outcomes retain the literal production/comparator identities and complete
planned dependency closure. Final-source transfer and H_final remain separate.

## Selected-delivery v2 resource closure

The explicit `reader.verify_selected_delivery_v2` entry point accepts six ordered
comparison and transfer graphs (B1, R1, B2, A1, A3, T1), then verifies A2 through
an exact `a2_closure` graph. It never dispatches A2 to `verify_a2_comparison`.
That graph pins the Option A human decision and its external seal, the v1 and v2
STOP-2 predecessor records and preserved seals, the raw campaign resource-ledger
snapshot, and the existing A2 reference and preflight artifacts.

The reference branch binds the original `gen1-stochastic-reference-admission-v1`
and root receipt to their native artifact map and canonical pin map. It runs the
offline native-reference verifier over the sealed 84-key set and requires its
summary to equal both existing admission projections. This validates retained
reference identity and coverage; it creates no new reference admission, execution
or comparison.

The preflight branch checks `PASS_NO_COMPUTATION`, zero declared production API
calls, the completed metadata snapshot, and exact candidate/reference pin links.
It raw-reads each direct `{path, sha256}` member of
`snapshot.candidate_artifacts` as finite supplemental history. Those bytes are
not parsed for authorization or acceptance. Other older snapshot references,
including command and completion metadata, are retained in the pinned snapshot
but are not traversed. This path does not replay preparation or resource-accounting
work.
The snapshot's aggregate and remaining-time fields are finite operational metadata
only. The profile does not calculate or report a conservative remaining budget
from them. It hashes the campaign ledger as opaque preserved bytes and does not
replay its charges.

V2 reports exactly six accepted family graphs. Its separate A2 closure summary
retains `PARTIAL` / `RESOURCE_LIMITED`, `PRODUCTION_COMPARISON_NOT_RUN`, production
acceptance and conservative remaining budget `NOT_ESTABLISHED`, budget exhaustion
`NOT_PROVEN`, and conditioning and physics failures `NOT_CLAIMED`. It has no A2
comparison, production P, transfer, result admission, arithmetic summary, or
accepted-family credit. The `A2_RESOURCE_LIMITED_CLOSURE_VERIFIED` status means
the software checked the retained byte graph; it is not a scientific acceptance.
