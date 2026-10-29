"""Independent seismic runner entry point; formal execution is HOLD.

Coordinator must first supply frozen relevant F and exact clean E containing
reviewed generator, runner and sealer. This initial scaffold cannot generate
formal data. Selfchecks live in tests/test_gen1_stochastic_seismic_oracle.py.
"""

import argparse
from typing import NamedTuple


class UnitCandidate(NamedTuple):
    """Internal result, never a frozen schema record or an admission decision."""

    key: tuple[str, ...]
    candidate: object
    error_type: str | None = None
    error_message: str | None = None
    admitted: bool = False


def _unit_key(key):
    # The seven-part tuple comes from the coordinator. Field spelling and
    # decimal canonicalization remain the responsibility of the future adapter.
    if not isinstance(key, tuple) or len(key) != 7:
        raise ValueError("seven-part exact key tuple required")
    if not all(isinstance(part, str) for part in key):
        raise TypeError("exact key fields must be strings")
    return key


def prepare_unit_candidates(fixtures, *, expected_keys, key_of, compute):
    """Parameter-only scaffold, no fixture/schema field names or policy guessed.

    key_of reads a unit fixture through a supplied adapter. compute invokes
    independent reference mathematics. Full expected-key preflight precedes
    every calculation. Ordinary calculation exceptions and unknown results are
    retained, with no invented zero uncertainty. Interruptions propagate to
    the future external supervisor. Successful computation is NOT admission.
    No output is serialized or persisted, and the formal CLI remains HOLD.
    Future frozen convergence/root/precision gates and schema serialization
    must be integrated only after root supplies the reviewed A snapshot.
    """
    expected = [_unit_key(key) for key in expected_keys]
    if len(set(expected)) != len(expected):
        raise ValueError("duplicate expected key")
    keyed = [(_unit_key(key_of(fixture)), fixture) for fixture in fixtures]
    actual = [key for key, _fixture in keyed]
    if len(set(actual)) != len(actual):
        raise ValueError("duplicate fixture key")
    if set(actual) != set(expected):
        raise ValueError("fixture key set differs from expected key set")
    outcomes = []
    for key, fixture in keyed:
        try:
            candidate = compute(fixture)
        except Exception as error:
            outcomes.append(UnitCandidate(key, None, type(error).__name__, str(error)))
        else:
            if candidate is None:
                outcomes.append(
                    UnitCandidate(
                        key,
                        None,
                        "UnknownCandidate",
                        "unknown independent reference result",
                    )
                )
            else:
                outcomes.append(UnitCandidate(key, candidate))
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal", action="store_true")
    parser.parse_args()
    parser.error("HOLD: frozen F, reviewed runner/sealer and clean E admission pending")


if __name__ == "__main__":
    main()
