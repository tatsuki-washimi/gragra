"""F_common_v2 frozen helpers: purpose, input hash, durability, identity.

Independent protocol helpers for the Gen1 stochastic common freeze v2
(``gen1-stochastic-F_common-v2``). All checks operate on synthetic
record/commit/hash fixtures only. This module imports no production code:
only the standard library plus the frozen v1 admission arithmetic
(``oracles.gen1.stochastic_acceptance``), which itself imports no
production code. No gragra import, no numerical reference generation, no
production output, and no production comparison are involved.

Frozen rule summary (normative text lives in
``docs/design/gen1/stochastic/common-freeze-v2.json``):

- R1: every calibration record and every formal science record carries an
  explicit ``purpose`` in {calibration, formal-science} plus ``set``
  membership in {calibration, science, auxiliary}. Missing, empty, or
  unknown purpose fails admission. Purpose ``calibration`` requires set
  ``calibration``; purpose ``formal-science`` requires set ``science`` or
  ``auxiliary``. Record keys reuse the v1 ``KEY_FIELDS`` and v1 key
  vocabulary verbatim (key ``set`` stays in {science, auxiliary}): the
  v1 ``set`` is never overloaded, so a calibration record and a formal
  science record can share an exact frozen key, and any such sharing
  fails admission.
- R2: each record defines a canonical scientific-input hash over the exact
  ordered preimage field list ``SCIENTIFIC_INPUT_PREIMAGE_FIELDS`` with
  canonical decimal/JSON encoding per the common-policy serialization
  (no floats anywhere). No calibration input hash may equal any formal
  science input hash, even under differing surface keys.
- R3: formal generation may start only when the exact E_family commit is
  an ancestor of the integration lineage pushed to PR #159 without
  history rewriting, or is preserved in a content-addressed Git
  bundle/archive with recorded SHA256 digest, successful ``git bundle
  verify``, and a restore procedure reproducing the exact E tree hash.
  Ephemeral-worktree-only commits are inadmissible; timestamp ordering
  alone never admits. History rewriting that makes a sealed E commit
  unreachable as an ancestor voids admission (fresh E, fresh attempt,
  fresh R required).
- R4: the normative common artifact set (exactly the 7 Gate-1 paths in
  ``NORMATIVE_COMMON_PATHS``) must be byte-identical across the protocol
  and integration lineages, verified per-path by SHA256. Any byte
  difference blocks formal generation until a new reviewed common freeze
  admits the changed bytes and the taint matrix is rechecked.
"""

import hashlib
import re

from oracles.gen1 import stochastic_acceptance as sa

PURPOSE_VOCABULARY = ("calibration", "formal-science")
SET_VOCABULARY = ("calibration", "science", "auxiliary")
PURPOSE_RECORD_FIELDS = {"purpose", "set", "key", "scientific_input"}
# R2 exact ordered preimage field list: decimal frequency, frame, PSD
# convention, source pins, procedure hash, frozen physical inputs.
SCIENTIFIC_INPUT_PREIMAGE_FIELDS = (
    "frequency_hz",
    "frame",
    "psd_convention",
    "source_pins",
    "procedure_hash",
    "inputs",
)
SOURCE_PIN_FIELDS = {"sha256", "equations", "pages"}
# R4 exact normative set: identical to the Gate-1
# normative_common_artifacts.paths (gate1-readonly-20261003.json).
NORMATIVE_COMMON_PATHS = (
    "docs/design/gen1/stochastic/B1-budget.json",
    "docs/design/gen1/stochastic/B1-fixtures.json",
    "docs/design/gen1/stochastic/B1-procedure.json",
    "docs/design/gen1/stochastic/common-freeze.json",
    "docs/design/gen1/stochastic/common-policy.json",
    "tests/oracles/gen1/stochastic_acceptance.py",
    "tests/test_gen1_stochastic_acceptance.py",
)
BUNDLE_RECORD_FIELDS = {
    "bundle_sha256",
    "verify_ok",
    "restored_tree_hash",
    "expected_tree_hash",
}
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")


def validate_purpose_record(record):
    """Validate purpose/set metadata; return the (purpose, set) pair."""
    if not isinstance(record, dict) or set(record) != PURPOSE_RECORD_FIELDS:
        raise ValueError(
            "purpose record requires exactly {purpose, set, key, scientific_input}"
        )
    purpose, membership = record["purpose"], record["set"]
    if not isinstance(purpose, str) or not purpose:
        raise ValueError("missing or empty purpose fails admission")
    if purpose not in PURPOSE_VOCABULARY:
        raise ValueError("unknown purpose fails admission")
    if membership not in SET_VOCABULARY:
        raise ValueError("unknown set membership fails admission")
    if (purpose == "calibration") != (membership == "calibration"):
        raise ValueError("purpose/set mapping violated")
    return purpose, membership


def v2_key_tuple(key):
    """Exact frozen key tuple: v1 fields and v1 vocabulary verbatim.

    Delegates to the frozen v1 ``key_tuple`` (key ``set`` stays in
    {science, auxiliary}). V2 adds no key fields; separation metadata
    lives in the top-level purpose/set fields only.
    """
    return sa.key_tuple(key)


def canonical_scientific_input_hash(scientific_input):
    """SHA256 over the exact ordered preimage under canonical encoding."""
    if not isinstance(scientific_input, dict) or set(scientific_input) != set(
        SCIENTIFIC_INPUT_PREIMAGE_FIELDS
    ):
        raise ValueError("scientific input requires exactly the frozen preimage fields")
    if sa.decimal_fraction(scientific_input["frequency_hz"]) <= 0:
        raise ValueError("positive frequency required")
    for field in ("frame", "psd_convention"):
        if (
            not isinstance(scientific_input[field], str)
            or not scientific_input[field].strip()
        ):
            raise ValueError("frame and PSD convention required")
    pins = scientific_input["source_pins"]
    if not isinstance(pins, list) or not pins:
        raise ValueError("source pins required")
    for pin in pins:
        if not isinstance(pin, dict) or set(pin) != SOURCE_PIN_FIELDS:
            raise ValueError("unknown source pin fields")
        if not _HEX64.fullmatch(pin["sha256"] or ""):
            raise ValueError("source SHA256 required")
        if not pin["equations"] or not pin["pages"]:
            raise ValueError("source equations/pages required")
    if not _HEX64.fullmatch(scientific_input["procedure_hash"] or ""):
        raise ValueError("procedure SHA256 required")
    if (
        not isinstance(scientific_input["inputs"], dict)
        or not scientific_input["inputs"]
    ):
        raise ValueError("complete frozen physical inputs required")
    preimage = {
        field: scientific_input[field] for field in SCIENTIFIC_INPUT_PREIMAGE_FIELDS
    }
    # canonical_json rejects floats, sorts keys, appends one LF; SHA256 is
    # taken over exactly those bytes.
    return hashlib.sha256(sa.canonical_json(preimage)).hexdigest()


def check_calibration_formal_separation(calibration_records, formal_records):
    """Admission gate: disjoint keys and disjoint canonical input hashes."""
    if not isinstance(calibration_records, list) or not isinstance(
        formal_records, list
    ):
        raise ValueError("record sets must be lists")
    cross = []
    for index, record in enumerate(calibration_records):
        if validate_purpose_record(record)[0] != "calibration":
            cross.append(("calibration", index))
    for index, record in enumerate(formal_records):
        if validate_purpose_record(record)[0] != "formal-science":
            cross.append(("formal-science", index))
    if cross:
        return {
            "admitted": False,
            "reason": "cross_purpose_record",
            "offending": cross,
            "shared_keys": [],
            "shared_hashes": [],
        }
    cal_keys = [v2_key_tuple(record["key"]) for record in calibration_records]
    formal_keys = [v2_key_tuple(record["key"]) for record in formal_records]
    shared_keys = sorted(set(cal_keys) & set(formal_keys))
    if shared_keys:
        return {
            "admitted": False,
            "reason": "key_overlap",
            "shared_keys": [list(key) for key in shared_keys],
            "shared_hashes": [],
        }
    cal_hashes = [
        canonical_scientific_input_hash(record["scientific_input"])
        for record in calibration_records
    ]
    formal_hashes = [
        canonical_scientific_input_hash(record["scientific_input"])
        for record in formal_records
    ]
    shared_hashes = sorted(set(cal_hashes) & set(formal_hashes))
    if shared_hashes:
        return {
            "admitted": False,
            "reason": "input_hash_overlap",
            "shared_keys": [],
            "shared_hashes": shared_hashes,
        }
    return {
        "admitted": True,
        "reason": "separate",
        "shared_keys": [],
        "shared_hashes": [],
        "calibration_count": len(calibration_records),
        "formal_count": len(formal_records),
    }


def acceptance_evidence_records(records):
    """Accept only formal-science records as acceptance evidence."""
    if not isinstance(records, list) or not records:
        raise ValueError("nonempty formal-science evidence set required")
    for record in records:
        if validate_purpose_record(record)[0] != "formal-science":
            raise ValueError("calibration evidence never creates acceptance evidence")
    return records


def check_e_durability(
    e_commit_sha,
    *,
    is_ancestor_now,
    was_ancestor_at_seal=True,
    bundle=None,
    timestamp_only=False,
):
    """Durability gate for the exact E_family commit.

    Returns a verdict dict with verdict in {admitted, blocked, void}:
    admitted via pushed ancestry or via a content-addressed bundle whose
    recorded SHA256 digest verifies and whose restore procedure reproduces
    the exact E tree hash; void when sealed ancestry was later destroyed
    by history rewriting; blocked otherwise (ephemeral worktree commits,
    dropped/rewritten SHAs presented without a bundle, timestamp-only
    claims).
    """
    if not isinstance(e_commit_sha, str) or not _HEX40.fullmatch(e_commit_sha):
        raise ValueError("exact E commit SHA required")
    mechanism = "pushed_ancestry" if is_ancestor_now else None
    bundle_valid = False
    if bundle is not None:
        if not isinstance(bundle, dict) or set(bundle) != BUNDLE_RECORD_FIELDS:
            raise ValueError("bundle record requires exactly the frozen bundle fields")
        if not _HEX64.fullmatch(bundle["bundle_sha256"] or ""):
            raise ValueError("bundle SHA256 digest required")
        if type(bundle["verify_ok"]) is not bool:
            raise ValueError("bundle verify flag requires an explicit boolean")
        for field in ("restored_tree_hash", "expected_tree_hash"):
            if not _HEX40.fullmatch(bundle[field] or ""):
                raise ValueError("exact E tree hashes required")
        bundle_valid = bool(
            bundle["verify_ok"]
            and bundle["restored_tree_hash"] == bundle["expected_tree_hash"]
        )
    if bundle_valid:
        mechanism = "content_addressed_bundle"
        return {
            "verdict": "admitted",
            "mechanism": mechanism,
            "e_commit_sha": e_commit_sha,
        }
    if was_ancestor_at_seal and not is_ancestor_now:
        return {
            "verdict": "void",
            "mechanism": "history_rewritten_after_seal",
            "e_commit_sha": e_commit_sha,
            "remedy": "fresh reviewed E, fresh attempt, fresh R required",
        }
    if is_ancestor_now:
        return {
            "verdict": "admitted",
            "mechanism": "pushed_ancestry",
            "e_commit_sha": e_commit_sha,
        }
    if timestamp_only:
        mechanism = "timestamp_only_insufficient"
    return {
        "verdict": "blocked",
        "mechanism": mechanism or "ephemeral_or_unpushed",
        "e_commit_sha": e_commit_sha,
    }


def check_normative_byte_identity(protocol_hashes, integration_hashes):
    """Byte-identity gate over exactly the 7 normative common paths."""
    for mapping in (protocol_hashes, integration_hashes):
        if not isinstance(mapping, dict) or set(mapping) != set(NORMATIVE_COMMON_PATHS):
            raise ValueError("hash maps must cover exactly the 7 normative paths")
        for digest in mapping.values():
            if not isinstance(digest, str) or not _HEX64.fullmatch(digest):
                raise ValueError("per-path SHA256 required")
    mismatched = sorted(
        path
        for path in NORMATIVE_COMMON_PATHS
        if protocol_hashes[path] != integration_hashes[path]
    )
    result = {
        "passed": not mismatched,
        "mismatched": mismatched,
        "checked_count": len(NORMATIVE_COMMON_PATHS),
    }
    if mismatched:
        result["remedy"] = (
            "formal generation BLOCKED until a new reviewed common freeze "
            "admits the changed bytes and the taint matrix is rechecked"
        )
    return result
