"""Versioned A2 reader seam: source binding vs runtime connection gates.

``gen1-stochastic-a2-reader-contract-v2`` separates:

- **Source model binding** (preflight / ``require_source_admission``):
  science ``claim_attestations`` must match ``SOURCE_BINDING_SCIENCE_CLAIMS``
  only (``SOURCE_MODEL_BINDING_ONLY`` scope). The production P377 packet uses
  an empty claim set; pins and proof reports carry binding evidence.

- **Runtime connection gates** (``family_connections.assess``):
  ``RUNTIME_CONNECTION_GATES`` are established from frozen procedure fields and
  live comparator evidence, not from source-review claim names.

Version identity (Major-2 evidence design, no frozen policy change):

1. Semantic contract version: ``CONTRACT_ID`` (this module).
2. Reader source identity: SHA-256 pins of versioned modules in the formal
   ``dependency_closure`` (includes ``reader_contract`` via ``inventory``).
3. Correspondence: wave-2 design note and PR evidence record
   ``CONTRACT_ID`` together with closure module pins; v1 formal admission
   artifacts are **not** auto-inherited onto v2 without a new admission bundle.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from ...common.transport import GateError

CONTRACT_ID = "gen1-stochastic-a2-reader-contract-v2"

# Frozen runtime obligations (also default when procedure omits explicit list).
RUNTIME_CONNECTION_GATES: tuple[str, ...] = (
    "numerical_procedure",
    "actual_x_transport",
    "SI",
    "coverage",
)

# Science-role claims allowed at source admission (empty for P377 companions).
SOURCE_BINDING_SCIENCE_CLAIMS: frozenset[str] = frozenset()


def validate_a2_runtime_connection_gates(runtime_requirements: Sequence[str]) -> None:
    """Ensure frozen procedure gates match the v2 runtime tuple exactly."""
    if tuple(runtime_requirements) != RUNTIME_CONNECTION_GATES:
        raise GateError("A2 runtime connection gates differ from reader contract")


def resolve_source_binding_requirements(
    family: str, runtime_requirements: Sequence[str]
) -> tuple[str, ...]:
    """Map runtime obligations to science claim names for source admission."""
    if family == "A2":
        validate_a2_runtime_connection_gates(runtime_requirements)
        return tuple(SOURCE_BINDING_SCIENCE_CLAIMS)
    return tuple(runtime_requirements)


def validate_science_binding_claim_keys(claim_keys: Iterable[str]) -> None:
    """Reject science attestations that encode runtime connection gate names."""
    if set(claim_keys) != set(SOURCE_BINDING_SCIENCE_CLAIMS):
        raise GateError("A2 source binding claims must not encode runtime gates")
