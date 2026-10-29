"""Complete planned typed closure equality, separate from source revision identity."""

from __future__ import annotations

import copy
import re
from pathlib import PurePosixPath

from .archive import EvidenceError, validate_pin

FIELDS = {
    "schema",
    "role",
    "production_commit",
    "production_tree",
    "modules",
    "import_edges",
    "packages",
    "native_backends",
    "artifacts",
    "python",
    "protected_packages",
}
IDENTITY_FIELDS = {"production_commit", "production_tree"}


def _validate(manifest: dict):
    if (
        set(manifest) != FIELDS
        or manifest["schema"] != "gen1-stochastic-dependencies-v1"
        or manifest["role"] not in {"production", "reference"}
    ):
        raise EvidenceError("unknown dependency closure schema/role")
    if any(
        not re.fullmatch(r"[0-9a-f]{40}", manifest[field]) for field in IDENTITY_FIELDS
    ):
        raise EvidenceError("actual immutable closure identity required")
    for field in (
        "modules",
        "import_edges",
        "packages",
        "native_backends",
        "artifacts",
        "protected_packages",
    ):
        if not isinstance(manifest[field], list):
            raise EvidenceError("typed closure domain must be a list")
    if not manifest["modules"] or not manifest["protected_packages"]:
        raise EvidenceError("nonempty complete planned sources required")
    fields = {
        "name",
        "package",
        "path",
        "root",
        "relative_path",
        "git_blob",
        "git_blob_missing_reason",
        "sha256",
        "kind",
    }
    names = []
    for module in manifest["modules"]:
        if (
            set(module) != fields
            or module["package"] not in manifest["protected_packages"]
        ):
            raise EvidenceError("unknown planned module schema/package")
        validate_pin({n: module[n] for n in ("path", "sha256", "kind")})
        try:
            relative = str(PurePosixPath(module["path"]).relative_to(module["root"]))
        except ValueError as error:
            raise EvidenceError("planned module outside declared root") from error
        if relative != module["relative_path"]:
            raise EvidenceError("scientific relative origin differs")
        if module["git_blob"] is None and not module["git_blob_missing_reason"]:
            raise EvidenceError("unexplained source provenance gap")
        if (
            module["git_blob"] is not None
            and module["git_blob_missing_reason"] is not None
        ):
            raise EvidenceError("source blob and missing reason contradict")
        names.append(module["name"])
    if len(names) != len(set(names)):
        raise EvidenceError("duplicate planned module")
    if len(manifest["protected_packages"]) != len(set(manifest["protected_packages"])):
        raise EvidenceError("duplicate protected package")
    edges = []
    for edge in manifest["import_edges"]:
        if set(edge) != {"from", "to"}:
            raise EvidenceError("unknown import edge fields")
        edges.append((edge["from"], edge["to"]))
    if len(edges) != len(set(edges)):
        raise EvidenceError("duplicate planned edge")
    for package in manifest["packages"]:
        if set(package) != {
            "name",
            "version",
            "location",
            "artifact",
            "artifact_missing_reason",
        }:
            raise EvidenceError("unknown package/build provenance schema")
        if package["artifact"] is None and not package["artifact_missing_reason"]:
            raise EvidenceError("unexplained package retrieval gap")
        if package["artifact"] is not None:
            validate_pin(package["artifact"])
            if package["artifact_missing_reason"] is not None:
                raise EvidenceError("package artifact and missing reason contradict")
    for artifact in manifest["native_backends"] + manifest["artifacts"]:
        validate_pin(artifact)
    if set(manifest["python"]) != {"implementation", "version", "executable"}:
        raise EvidenceError("unknown Python/build provenance schema")
    validate_pin(manifest["python"]["executable"])


def _normal_payload(manifest: dict, relocation: list[dict]) -> dict:
    value = copy.deepcopy(manifest)
    for field in IDENTITY_FIELDS:
        del value[field]
    mappings = []
    for entry in relocation:
        if set(entry) != {"from", "to"} or any(
            not isinstance(entry[n], str) or not PurePosixPath(entry[n]).is_absolute()
            for n in ("from", "to")
        ):
            raise EvidenceError("relocation accepts explicit absolute root pairs only")
        mappings.append((PurePosixPath(entry["from"]), PurePosixPath(entry["to"])))
    for index, (source, target) in enumerate(mappings):
        for other_source, other_target in mappings[index + 1 :]:
            if (
                source.is_relative_to(other_source)
                or other_source.is_relative_to(source)
                or target.is_relative_to(other_target)
                or other_target.is_relative_to(target)
            ):
                raise EvidenceError("ambiguous overlapping relocation roots")

    def path(original):
        current = PurePosixPath(original)
        for source, target in mappings:
            if current.is_relative_to(source):
                return str(target / current.relative_to(source))
        return original

    # Normalize only explicitly typed origins. Never rewrite relative paths,
    # payload strings, source hashes, versions, missing reasons, or import edges.
    for module in value["modules"]:
        module["path"], module["root"] = path(module["path"]), path(module["root"])
    for package in value["packages"]:
        package["location"] = path(package["location"])
        if package["artifact"]:
            package["artifact"]["path"] = path(package["artifact"]["path"])
    for pin in (
        value["native_backends"] + value["artifacts"] + [value["python"]["executable"]]
    ):
        pin["path"] = path(pin["path"])
    # Ordering carries no scientific meaning but cardinality/duplicates do.
    for field, key in (
        ("modules", lambda row: row["name"]),
        ("packages", lambda row: row["name"]),
        ("import_edges", lambda row: (row["from"], row["to"])),
        ("native_backends", lambda row: row["path"]),
        ("artifacts", lambda row: (row["path"], row["sha256"], row["kind"])),
    ):
        value[field] = sorted(value[field], key=key)
    value["protected_packages"] = sorted(value["protected_packages"])
    return value


def compare_closures(source: dict, target: dict, relocation: list[dict]) -> dict:
    """Compare all planned dependencies, including modules absent at runtime.

    The caller must separately verify independent review of relocation roots,
    raw dependency retrieval and Git byte identities at both revisions.
    Revision identity is retained, never replaced by an equivalence claim.
    """
    _validate(source)
    _validate(target)
    if _normal_payload(source, relocation) != _normal_payload(target, []):
        raise EvidenceError("complete typed dependency payloads are different")
    return {
        "status": "MATCH_COMPLETE_TYPED_PAYLOAD",
        "source_identity": {n: source[n] for n in sorted(IDENTITY_FIELDS)},
        "target_identity": {n: target[n] for n in sorted(IDENTITY_FIELDS)},
        "planned_module_count": len(source["modules"]),
        "observed_subset_used": False,
    }
