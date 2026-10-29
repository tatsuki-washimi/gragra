# Offline selected evidence reader

This candidate reads explicit raw artifacts without production imports, scientific
oracle execution, Git spawning or changes to execution originals. It assigns no
final source or delivery identity. CAS bytes preserve the original absolute pins.

`archive_selected` requires a fresh destination and explicit entries
`{original: {path, sha256, kind}, role}`. `EvidenceStore` resolves the original pin
through its retrieval map and hashes the bytes on every read. Sealed JSON paths
are never rewritten.

Comparison stage fields are phase, policy, candidate, authorization,
authorization_reviews, attempt, resource_supervisor, result_admission,
result_reviews, local_validator_proof, excluded_reviewer_ids and input_authorship. Result reviews
retain the existing nine-field gen1-stochastic-comparison-result-review-v1 schema.
Three different result reviewers must exclude the actual subject authors. Binding
reviews of earlier authority packets retain their distinct subject scope.
Authorization reviews approve the exact launch binding, so the root's explicit
policy permits an author of underlying physics or a plugin to review that
metadata. This exception does not apply to reference, production-source or actual
result reviews.

`input_authorship` contains exactly reference and production_source artifact pins.
Each uses gen1-stochastic-input-authorship-v1 with family, subject_kind, admission,
pins, authors and original_attestation. The original root-issued attestation uses
gen1-root-input-authorship-attestation-v1 with the same family/subject_kind/
admission/pins/authors plus issuer_id and issued_at. The projected and issued
rosters must match exactly; the issuer must not be an author. Author rows contain
reviewer_id and source_artifacts. Root records factual aliases of an actual author
explicitly, based on that review subject's implementation scope, rather than
inferring identities from Git email or a reviewer name. Earlier admissions are
retained, and only their own author roster excludes input reviewers.

The finite legacy exception applies only to R1 production source whose
pre-session authors remain unknown. It uses
gen1-stochastic-input-authorship-legacy-v1 with the ordinary input subjects,
authors (known session authors only), original_attestation,
historical_author_status=UNKNOWN_PRESESSION_OWNERSHIP_RETAINED and
reviewer_non_authorship (exact science/numerical/provenance pins). Its root original
uses gen1-root-input-authorship-legacy-attestation-v1 with matching fields plus
issuer_id/issued_at instead of original_attestation. No fictional historical
author is inserted. Every actual reviewer issues their own ten-field
gen1-stochastic-input-nonauthorship-review-v1 declaration: schema, family,
subject_kind, role, reviewer_id, admission, pins,
declaration=DID_NOT_AUTHOR_REVIEWED_IMPLEMENTATION, original_review and issued_at.
These bind the exact already-issued review identity/report/subject. Root cannot
replace the reviewer with itself. Known author IDs remain excluded. Other known
inputs retain their existing reviewed roster profile; this is not a general
waiver of scientific independence or a claim to know historical ownership.

Additive result-admission companions use schema
gen1-stochastic-result-admission-v1 with family, phase, decision, pins, attempt,
reviews and original_admission. Original receipts stay unchanged. Original local
validator proof companions use gen1-stochastic-local-validator-proof-v1 with
family, phase, profile, decision, subjects (policy/candidate/authorization/attempt/
resource_supervisor), validator (source pin/qualified_function), argv, pins and
original_proof. The original-environment result of the admitted mechanical
validator is required; reading its CAS bytes is not a claim of reexecution.

Complete planned closure equality includes unloaded sources, import edges,
packages, native libraries, configuration and Python. Root relocation changes
typed absolute origins only. Source and destination revision identities remain
separate. A transfer additionally needs all required archived dependency bytes,
literal reviewed package missing reasons, complete Git source catalogs and three
independent exact-subject transfer reviews. No observed subset proves equality.

Source catalogs use schema gen1-stochastic-source-catalog-v1 with commit, tree,
subtree (src/gragra), subtree_git_tree and files. Each file has relative_path,
mode, git_blob and artifact. Raw archived blobs reconstruct the entire Git subtree
hash; missing files fail. The original Git proof and transfer reviews bind this
catalog to actual source commits. The reader itself never spawns Git or infers
ancestry. Transfer reviews use gen1-stochastic-closure-transfer-review-v1 with
role, reviewer_id, decision, subjects, relocation and original_review.

The proposed selected-delivery packet uses
gen1-stochastic-selected-delivery-v1 with families, history and
final_identity_boundary (EXTERNAL_ATTESTATION_AFTER_TRACKED_COMMIT). Each family
entry has family/comparison/transfer, ordered B1,R1,B2,A1,A3,T1,A2. This is a
separate exact profile, not an arbitrary annotated index. History is retrieved
but never counted as accepted science. Unknown profiles fail closed.

The selected-comparison dispatcher uses the independently reviewed B1 profile
for its exact eleven-field connection/comparison stages. Its immutable accepted
legacy R_B1 authority remains separate from the reviewed R_aux records. B1 must
verify complete reviewed 70 before reading the 30-record science stage. Generic
families retain the twelve-field stage profile and replay the frozen arithmetic
only after actual result admission is verified. Returned summaries preserve
uncertainty claim kinds, science/AUX partitions, certified zero-budget reasons,
and the mandatory T1 x400 positive finite Gfree/SI gate.

A2 uses only the native `a2-ledger-v1` profile described in [A2-profile.md](A2-profile.md).
Its stage additionally binds the actual managed controller completion. A truthful
partial outcome requires a separate root-issued eligibility packet and the
ordinary three result reviews; it never counts as bounded scientific acceptance.
No actual A2 result is inferred from this source implementation. This candidate
does not claim complete seven-family delivery or final transfer verification.

The additive `gen1-stochastic-selected-delivery-v2` profile is entered explicitly
through `reader.verify_selected_delivery_v2`; the v1 reader remains strict and
unchanged. V2 carries exactly six accepted comparison/transfer graphs in B1, R1,
B2, A1, A3, T1 order. A2 is a separate `a2_closure` graph containing the human
Option A decision and seal, the preserved STOP-2 records/seals and campaign ledger
pin, the existing reference admission/root receipt/native artifact and canonical
pin maps, and the completed metadata preflight and snapshot. The native A2
reference verifier checks the retained 84-key graph (54 science, 30 auxiliary)
against its existing admission. No A2 production comparison, P identity, transfer,
result admission, partial-eligibility packet, or arithmetic is fabricated.

The A2 closure result is software evidence-graph verification only. It preserves
`PARTIAL` / `RESOURCE_LIMITED`, `PRODUCTION_COMPARISON_NOT_RUN`,
`NOT_ESTABLISHED` production acceptance and conservative remaining budget,
`NOT_PROVEN` budget exhaustion, and `NOT_CLAIMED` conditioning and physics
failures. The metadata preflight must say `PASS_NO_COMPUTATION`; its snapshot
must retain the exact candidate pin map and no-science scope. Snapshot time fields
are parsed only as finite operational metadata and are never interpreted as a
remaining-budget estimate. The previous campaign ledger is checked as opaque raw
bytes, and A2 remains outside the six accepted family count. V2 raw-reads each
direct `snapshot.candidate_artifacts` member as supplemental historical bytes and
reports their count; it does not parse them for authorization or acceptance. Other
snapshot references remain opaque metadata and are not traversed. The reader does
not replay preparation or resource-accounting work.

Transfer origin proofs use gen1-stochastic-git-subtree-proof-v1 with entries
commit/tree/subtree_git_tree/commit_object/root_tree_object/src_tree_object,
ordered source/destination catalogs. Raw Git hashing and tree links prove subtree
inclusion. Transfer descriptors also require authorship, using
gen1-stochastic-transfer-authorship-v1 with subjects, authors (each reviewer_id
and source_artifacts), and original_attestation. Those subject pins are reviewed
along with transfer. Authorship is not inferred from Git author names. The exact
accepted comparison closure pin is compulsory, independently of P commit labels.
Transfer original_attestation must parse as
gen1-root-transfer-authorship-attestation-v1 with subjects, authors, issuer_id and
issued_at matching the projected roster and exact subjects. Merely reading an
arbitrary raw report is insufficient.

Git cannot retain 0444 file mode. A fresh checkout's writable CAS is a transport
source and cannot be passed directly to EvidenceStore. Materialize its explicit
retrieval map into a fresh directory outside every Git ancestor first:

```python
from pathlib import Path
from tests.oracles.gen1.stochastic_delivery.archive import (
    decode_json, materialize_checkout, EvidenceStore,
)
mapping = decode_json(Path("retrieval-map.json").read_bytes())
report = materialize_checkout(Path("evidence"), mapping, Path("/var/tmp/fresh-evidence"))
store = EvidenceStore(Path(report["destination"]), mapping)
```

The materializer validates all selected raw SHA256 values before allocation,
copies bytes without touching original execution paths, chmods CAS objects 0444,
and rechecks every mapping through EvidenceStore. Symlink paths, existing
destinations, Git destinations and changed bytes fail. Its report describes byte
materialization only, never reexecution or scientific acceptance. Release
verification must exercise this step using an actual fresh checkout.
