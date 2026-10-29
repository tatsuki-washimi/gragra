"""Read-only constructors for the additive A2 v2 manifest fragments."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any


class ManifestError(ValueError):
    """The external diagnostic manifest differs from the frozen input contract."""


RETAINED_REFERENCE_SCHEMA = "gragra.a2.v2.retained-reference-manifest.v1"
RETAINED_ATTEMPT_ROOT = "/var/tmp/gragra-gen1-production-20261004/a2/attempts/a2-reference-001"
RETAINED_PINS = {
    "records": {
        "path": RETAINED_ATTEMPT_ROOT + "/records.raw.jsonl",
        "sha256": "1b30be2a4a1ebebd6b5baac1192c4e564ba4790cfcb299beff6e6bee930fe80a",
    },
    "seal": {
        "path": RETAINED_ATTEMPT_ROOT + "/seal.json",
        "sha256": "f9f788ce67901e7a4a23af64f69fd06f2b0ad452fd75f0a6568aad4b90b81ff5",
    },
    "admission": {
        "path": "/home/washimi/.local/state/gragra/gen1-production/20261004-pr159-final/a2/reference-admission-001-v1.json",
        "sha256": "e5bb957ce4abfb543f0ba0058962e325f8ce7f2461b1266a97730c47b35c58c8",
    },
}
FROZEN_ARTIFACT_PINS = {
    "fixture": (
        "docs/design/gen1/stochastic/A2-fixtures-v1.json",
        "8e5c5ba6f739bd978310b3087a12f966447d4e8e94427f14c776e6f4b1f020b7",
    ),
    "budget": (
        "docs/design/gen1/stochastic/A2-budget-v1.json",
        "21ae08e8056ab4ebbc645807a35a6ecbc2bdd4729c7d6b76fc70533f7d97ab81",
    ),
    "family_procedure": (
        "docs/design/gen1/stochastic/A2-procedure-v1.json",
        "c33ff1d2f6d9c811d1e229fbcc9d90d10bc17e0357dd2520f8cc8b83b6be5995",
    ),
    "freeze": (
        "docs/design/gen1/stochastic/A2-freeze-v1.json",
        "97042caf874364eaf6003f8dcdc4d160d07bb03ab91bb2e45d1df732ec2fb5a3",
    ),
}


def retained_reference_object(worktree_root: str | Path) -> dict[str, Any]:
    """Return the documented object for an external manifest without writing it."""
    root = Path(worktree_root)
    if not root.is_absolute():
        raise ManifestError("manifest paths.worktree_root must be absolute")
    try:
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise ManifestError("manifest worktree root is unavailable") from exc
    if not resolved.is_dir():
        raise ManifestError("manifest worktree root is not a directory")
    frozen = {
        name: {
            "path": str(resolved.joinpath(*PurePosixPath(relative).parts)),
            "sha256": digest,
        }
        for name, (relative, digest) in FROZEN_ARTIFACT_PINS.items()
    }
    return {
        "schema": RETAINED_REFERENCE_SCHEMA,
        "attempt_root": RETAINED_ATTEMPT_ROOT,
        "records": dict(RETAINED_PINS["records"]),
        "seal": dict(RETAINED_PINS["seal"]),
        "admission": dict(RETAINED_PINS["admission"]),
        "frozen_artifacts": frozen,
    }


def attach_retained_reference(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return an A2 root-manifest copy with the frozen retained-reference object."""
    if not isinstance(manifest, dict):
        raise ManifestError("external manifest must be an object")
    if (
        manifest.get("schema") != "gragra-diagnostic-manifest-v1"
        or manifest.get("model") != "A2"
        or manifest.get("diagnostic_only") is not True
        or type(manifest.get("science_credit")) is not int
        or manifest.get("science_credit") != 0
        or type(manifest.get("old_credit_seconds")) is not int
        or manifest.get("old_credit_seconds") != 0
    ):
        raise ManifestError("retained-reference attachment requires the fixed A2 diagnostic root")
    paths = manifest.get("paths")
    if not isinstance(paths, dict) or not isinstance(paths.get("worktree_root"), str):
        raise ManifestError("manifest paths.worktree_root is required")
    expected = retained_reference_object(paths["worktree_root"])
    existing = manifest.get("retained_reference")
    if existing is not None and existing != expected:
        raise ManifestError("existing manifest.retained_reference differs from the frozen pins")
    attached = dict(manifest)
    attached["retained_reference"] = expected
    return attached


def validate_retained_reference_object(manifest: dict[str, Any]) -> dict[str, Any]:
    """Require the exact retained object while preserving the family manifest schema."""
    if not isinstance(manifest, dict):
        raise ManifestError("external manifest must be an object")
    if (
        manifest.get("schema") != "gragra-diagnostic-manifest-v1"
        or manifest.get("model") != "A2"
        or manifest.get("diagnostic_only") is not True
        or type(manifest.get("science_credit")) is not int
        or manifest.get("science_credit") != 0
        or type(manifest.get("old_credit_seconds")) is not int
        or manifest.get("old_credit_seconds") != 0
    ):
        raise ManifestError("retained-reference validation requires the fixed A2 diagnostic root")
    paths = manifest.get("paths")
    if not isinstance(paths, dict) or not isinstance(paths.get("worktree_root"), str):
        raise ManifestError("manifest paths.worktree_root is required")
    expected = retained_reference_object(paths["worktree_root"])
    if manifest.get("retained_reference") != expected:
        raise ManifestError("manifest.retained_reference differs from the frozen artifact pins")
    return expected
