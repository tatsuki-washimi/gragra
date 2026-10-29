"""Content-addressed provenance contract for independently-derived oracle
JSON fixtures (``tests/oracles/data/*.json``).

Uses content-addressed checks instead of requiring a particular git history.
This keeps independently-derived scientific oracle data usable in an ordinary
checkout or a history-truncated copy without treating a historical commit as
part of the numerical contract.

The replacement is content-addressed: every claim this contract checks is a
SHA-256 comparison against bytes available in the working tree or the JSON
file itself, with no git history or network access required. Required
``provenance`` keys:

- ``oracle_name``: short identifier, must equal the caller's expected name.
- ``oracle_method_version``: human-readable version tag for the derivation
  *method* (independent of exact source bytes -- bump this when the
  mathematical approach changes, not on every edit).
- ``upstream_repository`` / ``upstream_revision``: declarative attribution
  only (a human-readable pointer to where this oracle was generated from).
  Neither is verified here -- ``upstream_revision`` in particular is NOT
  checked against git history by this contract (see
  ``tests/test_validation_extra_available.py`` for the separate, best-effort,
  self-skipping git-commit-existence audit kept for the internal repo only).
- ``source_content_sha256``: SHA-256 of the generator script's own source
  bytes, must match a fresh hash of the on-disk script.
- ``derivation_input_sha256``: SHA-256 of the generator's controlling
  constants/grids (a narrower signal than ``source_content_sha256``, which
  also changes on comment/docstring-only edits), must match a fresh hash
  computed by the caller from the generator module's current constants.
- ``golden_artifact_sha256``: SHA-256 of the JSON file's own non-provenance
  content, must match a fresh hash of that same content as currently loaded
  -- an integrity check against hand-editing/corruption of the golden data
  independent of the generator script.

Any missing key or any hash mismatch is fail-closed: this module raises
rather than returning a boolean, so a caller cannot accidentally ignore the
result the way an unchecked return value could be ignored.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REQUIRED_PROVENANCE_KEYS = (
    "oracle_name",
    "oracle_method_version",
    "upstream_repository",
    "upstream_revision",
    "source_content_sha256",
    "derivation_input_sha256",
    "golden_artifact_sha256",
)


class ProvenanceContractError(Exception):
    """Base class for content-addressed provenance contract failures."""


class MissingProvenanceError(ProvenanceContractError):
    """A required ``provenance`` key is absent from the oracle JSON."""


class ProvenanceDriftError(ProvenanceContractError):
    """A recorded provenance hash does not match a freshly computed one."""


def golden_artifact_sha256(non_provenance_body: dict) -> str:
    """SHA-256 of a canonical JSON encoding of an oracle's non-provenance
    content. Shared by the generator (computing the value to record) and
    this module's verification path (recomputing it from a loaded JSON file)
    so both sides serialize identically (``sort_keys=True``, compact
    separators) regardless of dict insertion order."""
    encoded = json.dumps(non_provenance_body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def verify_content_addressed_provenance(
    data: dict,
    *,
    oracle_name: str,
    generator_path: Path,
    derivation_input_sha256: str,
) -> None:
    """Fail-closed verification of an oracle JSON's content-addressed
    provenance block.

    ``derivation_input_sha256`` is supplied by the caller (computed by the
    generator module's own ``derivation_input_sha256()`` function against
    its *currently loaded* constants) rather than recomputed here, so this
    module does not need to import generator-specific internals.
    """
    provenance = data.get("provenance")
    if not isinstance(provenance, dict):
        raise MissingProvenanceError(
            f"{oracle_name}: oracle JSON has no 'provenance' object"
        )

    missing = [key for key in REQUIRED_PROVENANCE_KEYS if key not in provenance]
    if missing:
        raise MissingProvenanceError(
            f"{oracle_name}: provenance is missing required key(s) {missing} "
            "-- an undeclared/placeholder provenance field must never pass "
            "this gate silently."
        )

    if provenance["oracle_name"] != oracle_name:
        raise ProvenanceDriftError(
            f"expected oracle_name={oracle_name!r}, "
            f"found {provenance['oracle_name']!r} in the JSON"
        )

    actual_source_sha256 = hashlib.sha256(generator_path.read_bytes()).hexdigest()
    if provenance["source_content_sha256"] != actual_source_sha256:
        raise ProvenanceDriftError(
            f"{oracle_name}: source_content_sha256 mismatch -- "
            f"{generator_path.name} changed since the oracle JSON was "
            "generated. Rerun the generator and commit the updated JSON."
        )

    if provenance["derivation_input_sha256"] != derivation_input_sha256:
        raise ProvenanceDriftError(
            f"{oracle_name}: derivation_input_sha256 mismatch -- the "
            "generator's controlling constants/grids changed since the "
            "oracle JSON was generated. Rerun the generator and commit the "
            "updated JSON."
        )

    non_provenance_body = {k: v for k, v in data.items() if k != "provenance"}
    actual_golden_sha256 = golden_artifact_sha256(non_provenance_body)
    if provenance["golden_artifact_sha256"] != actual_golden_sha256:
        raise ProvenanceDriftError(
            f"{oracle_name}: golden_artifact_sha256 mismatch -- the oracle "
            "JSON's own golden data no longer matches its recorded hash "
            "(hand-edited or corrupted without regenerating)."
        )
