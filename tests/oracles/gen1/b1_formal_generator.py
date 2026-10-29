"""B1 formal generator adapter (Lane B, build-only HOLD).

Builds the formal-side plumbing WITHOUT launching any formal attempt: no
E_B1 construction, no R_B1, no P_science comparison, no sealing, no
publishing, no production read. Importing or calling this module starts
NOTHING; formal execution stays exclusively with the existing reviewed
runner (``run_stochastic_seismic_reference.prepare_unit_candidates``) and
a future reviewed sealer. There is no direct-publish path in this module.

What the adapter does (in memory only, no I/O besides read-only ``git
show`` of frozen objects):

* enumerates EXACTLY the 30 frozen B1 science fixture keys (verbatim copy
  cross-checked against the live frozen ``B1-fixtures.json`` content);
* verifies the F_B1_v2 / F_common_v2 pins (commit and content/blob hashes)
  before operating and fails closed on any mismatch;
* expands the 5 frozen aggregate science records into 30 per-key unit
  fixtures by exact-string reshuffling (no numeric conversion);
* requires generated-key coverage ``K_generated == K_expected`` with
  ``K_missing = K_unknown = K_duplicate`` all empty (fail closed);
* refuses calibration fixtures (fail closed, compute never invoked);
* refuses production imports and the forbidden mpmath reference path
  (fail closed if ``gragra*`` or ``stochastic_seismic_reference`` is
  already imported; this module never imports them);
* keeps every numerical-comparison diagnostic out of the formal path
  (there is no such route here at all);
* delegates execution through the existing runner and serializes resolved
  exact enclosures with the frozen serialization contract only
  (``serialize_enclosure`` / ``reparse`` / ``check_no_narrowing`` from the
  reviewed ``b1_analytic_enclosure`` module at its reviewed revision);
* retains unresolved cases as explicit records (never dropped) and keeps
  every candidate unadmitted.

Arithmetic note: this module performs no numerics at all; the only
mathematics reachable through it are the reviewed exact rational
primitives of ``b1_analytic_enclosure``. Frozen inputs are read ONLY via
read-only ``git show <pinned-commit>:<path>`` (content-addressed by pin);
working-tree copies are never trusted.
"""

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

GEN1_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Frozen pins (verbatim values, used as given inputs, never tuned)
# ---------------------------------------------------------------------------

#: F_B1_v2 freeze commit (protocol branch object, read-only).
_F_B1_V2_FREEZE_COMMIT = "e3a20eb276b779df7ff28a64e4c9ab278a1d0b65"
#: F_B1_v2 freeze commit, short form used for rev-parse cross-check.
_F_B1_V2_FREEZE_SHORT = "e3a20eb"
#: F_common_v2 freeze commit (protocol branch object, read-only).
_F_COMMON_V2_FREEZE_COMMIT = "ba9836ce03f714394a38a4dbe549c165bcc39152"
#: F_common_v2 freeze commit, short form.
_F_COMMON_V2_FREEZE_SHORT = "ba9836c"
#: Reviewed F_B1_v2 procedure-candidate commit (read-only).
_F_B1_V2_REVIEWED_COMMIT = "e5142f3f9f6647d125ed141fd30b59f78acc486e"
#: Reviewed F_B1_v2 procedure-candidate commit, short form.
_F_B1_V2_REVIEWED_SHORT = "e5142f3"

_FROZEN_DIR = "docs/design/gen1/stochastic"
_FROZEN_PROCEDURE_SHA256 = (
    "6e4aa87ff5d50cfdf1856e320aecfb847d283d9dd4307abaa2e247f73a340cd5"
)
_FROZEN_BUDGET_SHA256 = (
    "a3e321dcfd6c9825c2f7c330dbeb04c7dac0f970ff06464eb67d8cd53b79089c"
)
_FROZEN_FIXTURES_SHA256 = (
    "829e0e2cfd7aa915e19ad32f31abb0cdb57b0300d4854e5491f5358f0c197c05"
)
_FROZEN_COMMON_V2_SHA256 = (
    "2fb87ea29c65c0f6e6aae1c3ce083f6a544e14e8e90d720a35c0d4184c0a963d"
)

#: Reviewed enclosure-implementation blob (post-RC3 comment-only revision).
_REVIEWED_IMPL_BLOB = "4ffe0bb7c46bf47deea4e44cdc74ac9d7d087f66"
#: Reviewed enclosure-test blob.
_REVIEWED_TEST_BLOB = "9def277aa7a3eb1e4704b0bbd44d8f8c129b50b3"

_IMPL_FILE = GEN1_DIR / "b1_analytic_enclosure.py"
_RUNNER_FILE = GEN1_DIR / "run_stochastic_seismic_reference.py"
_REVIEWED_TEST_FILE = GEN1_DIR.parent.parent / "test_gen1_b1_analytic_enclosure.py"

#: Modules that must never be present on the formal path.
_FORBIDDEN_FORMAL_MODULE_TAILS = ("stochastic_seismic_reference",)

# ---------------------------------------------------------------------------
# The 30 frozen B1 science fixture keys (verbatim enumeration).
#
# Canonical key order per key:
#   (family, set, fixture_id, target_id, observable, representation,
#    frequency_hz)
# Representation spellings are the frozen verbatim forms ("Gfree"/"SI").
# ---------------------------------------------------------------------------

EXPECTED_B1_KEYS = (
    (
        "B1",
        "science",
        "rho2700_component",
        "origin",
        "acceleration_csd",
        "Gfree",
        "7e-1",
    ),
    ("B1", "science", "rho2700_component", "origin", "acceleration_csd", "SI", "7e-1"),
    (
        "B1",
        "science",
        "rho2700_component",
        "origin",
        "acceleration_csd",
        "Gfree",
        "3e+0",
    ),
    ("B1", "science", "rho2700_component", "origin", "acceleration_csd", "SI", "3e+0"),
    (
        "B1",
        "science",
        "rho2700_component",
        "origin",
        "acceleration_csd",
        "Gfree",
        "1.7e+1",
    ),
    (
        "B1",
        "science",
        "rho2700_component",
        "origin",
        "acceleration_csd",
        "SI",
        "1.7e+1",
    ),
    (
        "B1",
        "science",
        "rho1350_component",
        "origin",
        "acceleration_csd",
        "Gfree",
        "7e-1",
    ),
    ("B1", "science", "rho1350_component", "origin", "acceleration_csd", "SI", "7e-1"),
    (
        "B1",
        "science",
        "rho1350_component",
        "origin",
        "acceleration_csd",
        "Gfree",
        "3e+0",
    ),
    ("B1", "science", "rho1350_component", "origin", "acceleration_csd", "SI", "3e+0"),
    (
        "B1",
        "science",
        "rho1350_component",
        "origin",
        "acceleration_csd",
        "Gfree",
        "1.7e+1",
    ),
    (
        "B1",
        "science",
        "rho1350_component",
        "origin",
        "acceleration_csd",
        "SI",
        "1.7e+1",
    ),
    ("B1", "science", "rho2700_total", "origin", "acceleration_csd", "Gfree", "7e-1"),
    ("B1", "science", "rho2700_total", "origin", "acceleration_csd", "SI", "7e-1"),
    ("B1", "science", "rho2700_total", "origin", "acceleration_csd", "Gfree", "3e+0"),
    ("B1", "science", "rho2700_total", "origin", "acceleration_csd", "SI", "3e+0"),
    ("B1", "science", "rho2700_total", "origin", "acceleration_csd", "Gfree", "1.7e+1"),
    ("B1", "science", "rho2700_total", "origin", "acceleration_csd", "SI", "1.7e+1"),
    ("B1", "science", "zero_density", "origin", "acceleration_csd", "Gfree", "7e-1"),
    ("B1", "science", "zero_density", "origin", "acceleration_csd", "SI", "7e-1"),
    ("B1", "science", "zero_density", "origin", "acceleration_csd", "Gfree", "3e+0"),
    ("B1", "science", "zero_density", "origin", "acceleration_csd", "SI", "3e+0"),
    ("B1", "science", "zero_density", "origin", "acceleration_csd", "Gfree", "1.7e+1"),
    ("B1", "science", "zero_density", "origin", "acceleration_csd", "SI", "1.7e+1"),
    ("B1", "science", "zero_psd", "origin", "acceleration_csd", "Gfree", "7e-1"),
    ("B1", "science", "zero_psd", "origin", "acceleration_csd", "SI", "7e-1"),
    ("B1", "science", "zero_psd", "origin", "acceleration_csd", "Gfree", "3e+0"),
    ("B1", "science", "zero_psd", "origin", "acceleration_csd", "SI", "3e+0"),
    ("B1", "science", "zero_psd", "origin", "acceleration_csd", "Gfree", "1.7e+1"),
    ("B1", "science", "zero_psd", "origin", "acceleration_csd", "SI", "1.7e+1"),
)

EXPECTED_KEY_COUNT = 30

_KEY_FIELDS = (
    "family",
    "set",
    "fixture_id",
    "target_id",
    "observable",
    "representation",
    "frequency_hz",
)

_REPRESENTATIONS = ("Gfree", "SI")
_SCIENCE_OBSERVABLE = "acceleration_csd"


# ---------------------------------------------------------------------------
# Errors (all fail-closed; none is ever swallowed into a record)
# ---------------------------------------------------------------------------


class FormalGeneratorError(Exception):
    """Base error for the formal generator adapter."""


class FrozenPinMismatchError(FormalGeneratorError):
    """A frozen commit/content/blob pin did not verify."""


# Preserve the established exception API and serialized error names.
FrozenPinMismatch = FrozenPinMismatchError
FrozenPinMismatchError.__name__ = "FrozenPinMismatch"
FrozenPinMismatchError.__qualname__ = "FrozenPinMismatch"


class KeyBoundaryViolationError(FormalGeneratorError):
    """Generated key coverage differs from the frozen 30-key boundary."""


# Preserve the established exception API and serialized error names.
KeyBoundaryViolation = KeyBoundaryViolationError
KeyBoundaryViolationError.__name__ = "KeyBoundaryViolation"
KeyBoundaryViolationError.__qualname__ = "KeyBoundaryViolation"


class MissingKeyError(KeyBoundaryViolation):
    """Expected frozen keys absent from the generated set."""


class UnknownKeyError(KeyBoundaryViolation):
    """Generated keys outside the frozen expected set."""


class DuplicateKeyError(KeyBoundaryViolation):
    """Duplicate generated keys."""


class CalibrationContamination(KeyBoundaryViolation):
    """A calibration fixture reached the formal science path."""


class ProductionDeniedError(FormalGeneratorError):
    """A production import is present; the formal path refuses to operate."""


# Preserve the established exception API and serialized error names.
ProductionDenied = ProductionDeniedError
ProductionDeniedError.__name__ = "ProductionDenied"
ProductionDeniedError.__qualname__ = "ProductionDenied"


class ForbiddenPathError(FormalGeneratorError):
    """The forbidden mpmath reference path is present on the formal path."""


class InvalidEnclosureError(FormalGeneratorError):
    """A compute result is not an exact ordered Fraction enclosure."""


# ---------------------------------------------------------------------------
# Read-only frozen access (git show only; never the working tree)
# ---------------------------------------------------------------------------


def _git(*args):
    try:
        proc = subprocess.run(
            ["git", "-C", str(GEN1_DIR), *args],
            capture_output=True,
            text=False,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise FrozenPinMismatch(f"git unavailable: {error}") from error
    return proc


def _git_show(commit, path):
    proc = _git("show", f"{commit}:{path}")
    if proc.returncode != 0:
        raise FrozenPinMismatch(f"frozen object unavailable: {commit}:{path}")
    return proc.stdout


def _rev_parse_commit(short, full):
    proc = _git("rev-parse", f"{short}^{{commit}}")
    if proc.returncode != 0:
        raise FrozenPinMismatch(f"frozen commit unavailable: {short}")
    if proc.stdout.decode("ascii").strip() != full:
        raise FrozenPinMismatch(f"frozen commit mismatch: {short}")
    return full


def _hash_object(path):
    proc = _git("hash-object", str(path))
    if proc.returncode != 0:
        raise FrozenPinMismatch(f"blob unavailable: {path}")
    return proc.stdout.decode("ascii").strip()


def _check_content(commit, filename, expected_sha256):
    data = _git_show(commit, f"{_FROZEN_DIR}/{filename}")
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected_sha256:
        raise FrozenPinMismatch(f"frozen content mismatch: {commit}:{filename}")
    return data


def verify_frozen_pins():
    """Verify every F_B1_v2 / F_common_v2 pin; fail closed on mismatch.

    Checks the three frozen commit identities, the four frozen content
    hashes (procedure, budget, fixtures, common-v2), and the two reviewed
    blob hashes (enclosure implementation + enclosure tests). Returns a
    pin report mapping on success.
    """
    _rev_parse_commit(_F_B1_V2_FREEZE_SHORT, _F_B1_V2_FREEZE_COMMIT)
    _rev_parse_commit(_F_COMMON_V2_FREEZE_SHORT, _F_COMMON_V2_FREEZE_COMMIT)
    _rev_parse_commit(_F_B1_V2_REVIEWED_SHORT, _F_B1_V2_REVIEWED_COMMIT)
    _check_content(
        _F_B1_V2_REVIEWED_SHORT,
        "B1-procedure-v2.json",
        _FROZEN_PROCEDURE_SHA256,
    )
    _check_content(_F_B1_V2_REVIEWED_SHORT, "B1-budget.json", _FROZEN_BUDGET_SHA256)
    _check_content(_F_B1_V2_FREEZE_SHORT, "B1-fixtures.json", _FROZEN_FIXTURES_SHA256)
    _check_content(
        _F_COMMON_V2_FREEZE_SHORT,
        "common-freeze-v2.json",
        _FROZEN_COMMON_V2_SHA256,
    )
    impl_blob = _hash_object(_IMPL_FILE)
    if impl_blob != _REVIEWED_IMPL_BLOB:
        raise FrozenPinMismatch("reviewed enclosure-implementation blob mismatch")
    test_blob = _hash_object(_REVIEWED_TEST_FILE)
    if test_blob != _REVIEWED_TEST_BLOB:
        raise FrozenPinMismatch("reviewed enclosure-test blob mismatch")
    return {
        "f_b1_v2_freeze_commit": _F_B1_V2_FREEZE_COMMIT,
        "f_common_v2_freeze_commit": _F_COMMON_V2_FREEZE_COMMIT,
        "f_b1_v2_reviewed_commit": _F_B1_V2_REVIEWED_COMMIT,
        "procedure_sha256": _FROZEN_PROCEDURE_SHA256,
        "budget_sha256": _FROZEN_BUDGET_SHA256,
        "fixtures_sha256": _FROZEN_FIXTURES_SHA256,
        "common_v2_sha256": _FROZEN_COMMON_V2_SHA256,
        "reviewed_impl_blob": _REVIEWED_IMPL_BLOB,
        "reviewed_test_blob": _REVIEWED_TEST_BLOB,
    }


def load_frozen_fixtures():
    """Load the frozen B1 fixture document via read-only ``git show``.

    Pin verification runs first; any mismatch fails closed before the
    document is parsed. Working-tree copies are never read.
    """
    verify_frozen_pins()
    data = _git_show(_F_B1_V2_FREEZE_SHORT, f"{_FROZEN_DIR}/B1-fixtures.json")
    try:
        document = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise FrozenPinMismatch(f"frozen fixtures unparsable: {error}") from error
    if not isinstance(document, dict):
        raise FrozenPinMismatch("frozen fixtures document is not a mapping")
    return document


# ---------------------------------------------------------------------------
# Reviewed primitives + existing runner (loaded from file, never packaged)
# ---------------------------------------------------------------------------

_ENCLOSURE_MODULE = None
_RUNNER_MODULE = None


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise FrozenPinMismatch(f"module unavailable: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reviewed_primitives():
    """Return the reviewed enclosure module at exactly its reviewed blob."""
    global _ENCLOSURE_MODULE
    if _hash_object(_IMPL_FILE) != _REVIEWED_IMPL_BLOB:
        raise FrozenPinMismatch("reviewed enclosure-implementation blob mismatch")
    if _ENCLOSURE_MODULE is None:
        _ENCLOSURE_MODULE = _load_module("b1_analytic_enclosure", _IMPL_FILE)
    return _ENCLOSURE_MODULE


def existing_runner():
    """Return the existing reviewed runner scaffold module (unmodified)."""
    global _RUNNER_MODULE
    if _RUNNER_MODULE is None:
        _RUNNER_MODULE = _load_module("run_stochastic_seismic_reference", _RUNNER_FILE)
        if not hasattr(_RUNNER_MODULE, "prepare_unit_candidates"):
            raise FrozenPinMismatch("existing runner entry point missing")
    return _RUNNER_MODULE


# ---------------------------------------------------------------------------
# Formal-path purity (no production, no mpmath reference path)
# ---------------------------------------------------------------------------


def assert_formal_purity():
    """Fail closed if production or the forbidden reference path is loaded."""
    for name in sys.modules:
        if name == "gragra" or name.startswith("gragra."):
            raise ProductionDenied(f"production import present on formal path: {name}")
        if name.split(".")[-1] in _FORBIDDEN_FORMAL_MODULE_TAILS:
            raise ForbiddenPathError(f"forbidden mpmath reference path present: {name}")
    return True


# ---------------------------------------------------------------------------
# Key boundary: the exact 30 frozen science keys
# ---------------------------------------------------------------------------


def key_of(unit):
    """Canonical 7-tuple key of one per-key unit fixture (exact strings)."""
    if not isinstance(unit, dict):
        raise TypeError("unit fixture must be a mapping")
    try:
        parts = tuple(unit[field] for field in _KEY_FIELDS)
    except KeyError as error:
        raise ValueError(f"unit fixture missing key field: {error}") from error
    for part in parts:
        if not isinstance(part, str) or not part:
            raise TypeError("exact non-empty key strings required")
    return parts


def expected_b1_keys(document=None):
    """Return the 30 frozen science keys, cross-checked against the pins.

    The live frozen document's ``expected_keys`` must reproduce the
    verbatim :data:`EXPECTED_B1_KEYS` enumeration exactly (same count,
    same set); any drift fails closed.
    """
    if document is None:
        document = load_frozen_fixtures()
    try:
        raw = document["expected_keys"]
    except (KeyError, TypeError) as error:
        raise FrozenPinMismatch(f"frozen expected_keys unavailable: {error}") from error
    live = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise FrozenPinMismatch("frozen expected_keys entry not a mapping")
        try:
            live.append(tuple(entry[field] for field in _KEY_FIELDS))
        except KeyError as error:
            raise FrozenPinMismatch(
                f"frozen expected_keys entry missing field: {error}"
            ) from error
    if len(live) != EXPECTED_KEY_COUNT or set(live) != set(EXPECTED_B1_KEYS):
        raise FrozenPinMismatch("frozen expected_keys drift from verbatim pin")
    return list(EXPECTED_B1_KEYS)


def expand_science_fixtures(science_records):
    """Expand frozen aggregate science records to 30 per-key unit fixtures.

    Pure exact-string reshuffling: each aggregate record's parallel
    frequency / displacement lists become one unit per (frequency,
    representation) pair, carrying the exact decimal input strings for
    the reviewed-primitive compute. Returns exactly 30 units whose keys
    equal :data:`EXPECTED_B1_KEYS`.
    """
    units = []
    for record in science_records:
        if not isinstance(record, dict):
            raise FrozenPinMismatch("frozen science record not a mapping")
        try:
            fixture_id = record["fixture_id"]
            frequencies = record["frequency_hz"]
            spectra = record["displacement_psd_m2_hz"]
            density = record["density_kg_m3"]
            g_si = record["g_si"]
            target = record["target_id"]
        except KeyError as error:
            raise FrozenPinMismatch(
                f"frozen science record missing field: {error}"
            ) from error
        if len(frequencies) != len(spectra):
            raise FrozenPinMismatch("frozen science frequency/spectrum mismatch")
        for frequency, spectrum in zip(frequencies, spectra, strict=False):
            for representation in _REPRESENTATIONS:
                units.append(
                    {
                        "family": "B1",
                        "set": "science",
                        "fixture_id": fixture_id,
                        "target_id": target,
                        "observable": _SCIENCE_OBSERVABLE,
                        "representation": representation,
                        "frequency_hz": frequency,
                        "density_kg_m3": density,
                        "displacement_psd_m2_hz": spectrum,
                        "g_si": g_si,
                    }
                )
    if len(units) != EXPECTED_KEY_COUNT:
        raise FrozenPinMismatch("science expansion did not yield 30 units")
    if [key_of(unit) for unit in units] != list(EXPECTED_B1_KEYS):
        raise FrozenPinMismatch("science expansion keys drift from verbatim pin")
    return units


def audit_key_set(units, expected):
    """Audit generated-key coverage; return sets (no raise here)."""
    generated = [key_of(unit) for unit in units]
    expected_list = list(expected)
    counts = {}
    for key in generated:
        counts[key] = counts.get(key, 0) + 1
    duplicate = sorted(key for key, n in counts.items() if n > 1)
    expected_set = set(expected_list)
    generated_set = set(generated)
    return {
        "generated": generated,
        "expected": expected_list,
        "missing": sorted(expected_set - generated_set),
        "unknown": sorted(generated_set - expected_set),
        "duplicate": duplicate,
    }


def require_exact_coverage(units, expected):
    """Require K_generated == K_expected with empty missing/unknown/dup."""
    audit = audit_key_set(units, expected)
    if audit["duplicate"]:
        raise DuplicateKeyError(f"duplicate fixture keys: {audit['duplicate']}")
    if audit["unknown"]:
        raise UnknownKeyError(f"unknown fixture keys: {audit['unknown']}")
    if audit["missing"]:
        raise MissingKeyError(f"missing expected keys: {audit['missing']}")
    return audit


def refuse_calibration(units):
    """Fail closed if any calibration fixture reaches the science path."""
    for unit in units:
        if not isinstance(unit, dict):
            raise TypeError("unit fixture must be a mapping")
        fixture_id = unit.get("fixture_id", "")
        set_name = unit.get("set", "")
        if set_name != "science" or (
            isinstance(fixture_id, str) and fixture_id.startswith("calibration_")
        ):
            raise CalibrationContamination(
                "calibration fixture on formal science path: "
                f"{fixture_id!r} (set {set_name!r})"
            )
    return True


# ---------------------------------------------------------------------------
# Formal record construction (in memory only; no sealing, no publishing)
# ---------------------------------------------------------------------------


def decode_record_enclosure(record):
    """Reparse a resolved record and prove no serialization narrowing."""
    primitives = reviewed_primitives()
    if record.get("status") != "resolved":
        raise InvalidEnclosureError("only resolved records decode to enclosures")
    enclosure = record.get("enclosure")
    if not isinstance(enclosure, dict):
        raise InvalidEnclosureError("resolved record lacks enclosure mapping")
    primitives.check_no_narrowing(enclosure)
    return primitives.reparse(enclosure)


def build_formal_records(units, *, compute):
    """Build the 30 in-memory formal records through the existing runner.

    ``compute`` maps one unit fixture to an exact ``(lo, hi)`` Fraction
    enclosure, or to ``None`` when the key is (currently) unresolvable;
    raised errors also yield explicit unresolved records (never dropped).
    Every step fails closed: purity, pins, calibration refusal, exact
    30-key coverage, then delegation to the existing runner's
    ``prepare_unit_candidates``. Resolved candidates that are not exact
    ordered Fraction enclosures fail closed. All records stay unadmitted:
    this function seals nothing, publishes nothing, compares nothing,
    and writes no files.
    """
    assert_formal_purity()
    verify_frozen_pins()
    primitives = reviewed_primitives()
    runner = existing_runner()
    expected = expected_b1_keys()
    refuse_calibration(units)
    audit = require_exact_coverage(units, expected)

    def _compute(unit):
        return compute(unit)

    candidates = runner.prepare_unit_candidates(
        list(units),
        expected_keys=expected,
        key_of=key_of,
        compute=_compute,
    )
    if len(candidates) != EXPECTED_KEY_COUNT:
        raise FrozenPinMismatch("runner did not return all 30 candidates")
    records = []
    for candidate in candidates:
        if candidate.admitted:
            raise FormalGeneratorError("runner claimed admission; refused")
        if candidate.candidate is None:
            records.append(
                {
                    "key": candidate.key,
                    "status": "unresolved",
                    "error_type": candidate.error_type,
                    "error_message": candidate.error_message,
                }
            )
            continue
        try:
            lo, hi = candidate.candidate
        except (TypeError, ValueError) as error:
            raise InvalidEnclosureError(
                f"compute result not a (lo, hi) enclosure: {error}"
            ) from error
        from fractions import Fraction

        if not isinstance(lo, Fraction) or not isinstance(hi, Fraction) or not lo <= hi:
            raise InvalidEnclosureError("enclosure must be ordered Fractions")
        records.append(
            {
                "key": candidate.key,
                "status": "resolved",
                "enclosure": primitives.serialize_enclosure(lo, hi),
            }
        )
    if [record["key"] for record in records] != expected:
        raise FrozenPinMismatch("formal records dropped or reordered keys")
    return records, audit


# ---------------------------------------------------------------------------
# Generator revision: nonzero candidate construction through the reviewed
# enclosure primitives (build-only HOLD; no formal attempt launched here).
#
# This revision routes the 18 frozen nonzero science fixtures (the three
# rho* families) through EXACTLY the reviewed primitives of
# ``b1_analytic_enclosure`` at its reviewed blob (see
# :func:`verify_frozen_pins`): exact decimal parsing, the primary pi
# enclosure, directed scaling/squaring, the component/total normalization,
# the single-ownership uncertainty gate, the G1/G2 confusion guard, and the
# independent SI recomputation. The reviewed implementation file itself is
# NOT modified by this revision. Zero fixtures keep the exact prior
# behavior (the point enclosure). Malformed inputs raise honestly and are
# retained as unresolved records by the existing runner; they are never
# admitted and never narrow an enclosure. No production read, no reference
# path, no comparison, no sealing, no publishing anywhere in this section.
# ---------------------------------------------------------------------------

#: Frozen nonzero science fixture ids (verbatim spellings).
_NONZERO_FIXTURE_IDS = (
    "rho2700_component",
    "rho1350_component",
    "rho2700_total",
)

#: Frozen zero science fixture ids (verbatim spellings; path unchanged).
_ZERO_FIXTURE_IDS = ("zero_density", "zero_psd")

#: Frozen representation spellings mapped to the reviewed guard tags.
_REPRESENTATION_GUARD_TAGS = {"Gfree": "g_free", "SI": "si"}

#: Process-local candidate cache: reviewed primary pi enclosure plus the
#: frozen fixture convention map. Both entries are deterministic values
#: read from reviewed/frozen sources only.
_CANDIDATE_CACHE = {}


def clear_candidate_cache():
    """Drop cached candidate values (test-only helper; no formal effect)."""
    _CANDIDATE_CACHE.clear()


def convention_for_fixture(fixture_id):
    """Return the frozen ``input_convention`` for one science fixture id.

    Sourced from the frozen fixture document via read-only ``git show``
    (cached per process); unknown ids or conventions fail closed.
    """
    conventions = _CANDIDATE_CACHE.get("conventions")
    if conventions is None:
        document = load_frozen_fixtures()
        conventions = {}
        for record in document.get("science", []):
            if not isinstance(record, dict):
                raise FrozenPinMismatch("frozen science record not a mapping")
            try:
                conventions[record["fixture_id"]] = record["input_convention"]
            except KeyError as error:
                raise FrozenPinMismatch(
                    f"frozen science record missing field: {error}"
                ) from error
        _CANDIDATE_CACHE["conventions"] = conventions
    try:
        convention = conventions[fixture_id]
    except (KeyError, TypeError) as error:
        raise ValueError(f"unknown science fixture_id: {error}") from error
    if convention not in ("component", "total"):
        raise ValueError("unknown input convention on science path")
    return convention


def cached_primary_pi_enclosure():
    """Return the reviewed primary pi enclosure (cached per process)."""
    cached = _CANDIDATE_CACHE.get("pi_primary")
    if cached is None:
        primitives = reviewed_primitives()
        cached = primitives.pi_enclosure_primary()
        _CANDIDATE_CACHE["pi_primary"] = cached
    return cached


def reviewed_nonzero_compute(unit):
    """Build one exact ``(lo, hi)`` Fraction enclosure via reviewed steps.

    Zero-density and zero-PSD units take the exact prior path (the point
    enclosure). Every other frozen science unit is enclosed as the
    longitudinal scale ``(4 pi rho)^2 S`` (divided by 3 under the frozen
    ``total`` convention via the reviewed component/total conversion),
    with the pi factor owned once by the reviewed primary enclosure and
    the SI representation recomputed independently as ``G_SI^2`` times the
    exact G-free enclosure with its own (here zero) residual. Each step
    uses a reviewed primitive, so malformed or out-of-domain inputs raise
    from the reviewed checks and surface as retained unresolved records.
    """
    from fractions import Fraction

    primitives = reviewed_primitives()
    if not isinstance(unit, dict):
        raise TypeError("unit fixture must be a mapping")
    try:
        fixture_id = unit["fixture_id"]
        representation = unit["representation"]
        density_text = unit["density_kg_m3"]
        spectrum_text = unit["displacement_psd_m2_hz"]
        g_text = unit["g_si"]
    except KeyError as error:
        raise ValueError(f"unit fixture missing field: {error}") from error
    if representation not in _REPRESENTATION_GUARD_TAGS:
        raise ValueError("unknown representation on science path")
    tag = _REPRESENTATION_GUARD_TAGS[representation]
    rho = primitives.parse_decimal(density_text)
    spectrum = primitives.parse_decimal(spectrum_text)
    if rho < 0 or spectrum < 0:
        raise ValueError("nonnegative density and PSD required")
    if rho == 0 or spectrum == 0:
        # Exact-zero path, unchanged from the prior revision.
        return (Fraction(0), Fraction(0))
    convention = convention_for_fixture(fixture_id)
    pi_lo, pi_hi = cached_primary_pi_enclosure()
    if not Fraction(0) < pi_lo <= pi_hi:
        raise ValueError("reviewed pi enclosure not positive")
    transfer = primitives.multiply_enclosure(pi_lo, pi_hi, 4 * rho)
    gfree = primitives.square_enclosure(transfer[0], transfer[1], spectrum)
    primitives.verify_square_encloses(transfer[0], transfer[1], spectrum, gfree)
    if convention == "total":
        gfree = (
            primitives.component_total_convert(gfree[0], "to_component"),
            primitives.component_total_convert(gfree[1], "to_component"),
        )
    if not gfree[0] <= gfree[1]:
        raise ValueError("unordered G-free enclosure")
    primitives.allocation_nonzero_requires_uncertainty(spectrum, gfree[1] - gfree[0])
    primitives.g2_confusion_guard(tag, gfree[0])
    primitives.g2_confusion_guard(tag, gfree[1])
    if representation == "Gfree":
        return gfree
    si_lo, si_hi, _g = primitives.si_recompute(gfree[0], gfree[1], g_text, Fraction(0))
    return (si_lo, si_hi)


def nonzero_fixture_keys():
    """Return the 18 frozen nonzero science keys in canonical order."""
    return [key for key in EXPECTED_B1_KEYS if key[2] in _NONZERO_FIXTURE_IDS]


def zero_fixture_keys():
    """Return the 12 frozen zero science keys in canonical order."""
    return [key for key in EXPECTED_B1_KEYS if key[2] in _ZERO_FIXTURE_IDS]


def main(argv=None):
    """Formal CLI stays HOLD: the adapter alone launches nothing."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal", action="store_true")
    parser.parse_args(argv)
    parser.error(
        "HOLD: adapter builds plumbing only; formal launch requires the "
        "reviewed runner/sealer and clean E admission"
    )


if __name__ == "__main__":
    main()
