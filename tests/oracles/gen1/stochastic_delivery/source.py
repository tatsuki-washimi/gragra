"""Verify complete explicitly selected Git subtrees from archived raw bytes."""

from __future__ import annotations

import hashlib
from pathlib import PurePosixPath

from .archive import EvidenceError


def git_oid(kind, raw):
    return hashlib.sha1(
        kind.encode() + b" " + str(len(raw)).encode() + b"\0" + raw
    ).hexdigest()


def verify_catalog(store, catalog):
    fields = {"schema", "commit", "tree", "subtree", "subtree_git_tree", "files"}
    if (
        not isinstance(catalog, dict)
        or set(catalog) != fields
        or catalog["schema"] != "gen1-stochastic-source-catalog-v1"
        or catalog["subtree"] != "src/gragra"
        or not catalog["files"]
    ):
        raise EvidenceError("explicit complete gragra subtree catalog required")
    nodes = {}
    for row in catalog["files"]:
        if set(row) != {"relative_path", "mode", "git_blob", "artifact"} or row[
            "mode"
        ] not in {"100644", "100755"}:
            raise EvidenceError("unknown Git source entry/mode")
        path = PurePosixPath(row["relative_path"])
        if (
            path.is_absolute()
            or any(part in {".", ".."} for part in path.parts)
            or str(path) != row["relative_path"]
            or not path.parts
        ):
            raise EvidenceError("invalid scientific source relative path")
        if row["relative_path"] in nodes:
            raise EvidenceError("duplicate source catalog path")
        raw = store.read(row["artifact"])
        if git_oid("blob", raw) != row["git_blob"]:
            raise EvidenceError("archived raw source differs from Git blob")
        nodes[row["relative_path"]] = (row["mode"], row["git_blob"])

    def tree(prefix):
        files, directories = {}, set()
        for relative, identity in nodes.items():
            if not relative.startswith(prefix):
                continue
            suffix = relative[len(prefix) :]
            name, separator, _tail = suffix.partition("/")
            if separator:
                directories.add(name)
            else:
                files[name] = identity
        if set(files) & directories:
            raise EvidenceError("source path is both directory and blob")
        rows = [(name, mode, oid) for name, (mode, oid) in files.items()]
        rows += [(name, "40000", tree(prefix + name + "/")) for name in directories]
        rows.sort(
            key=lambda row: (row[0] + ("/" if row[1] == "40000" else "")).encode()
        )
        raw = b"".join(
            mode.encode() + b" " + name.encode() + b"\0" + bytes.fromhex(oid)
            for name, mode, oid in rows
        )
        return git_oid("tree", raw)

    if tree("") != catalog["subtree_git_tree"]:
        raise EvidenceError("incomplete/changed source subtree catalog")
    return nodes


def verify_closure_retrieval(store, closure):
    """Every planned source/native/config/interpreter must have exact CAS bytes.

    An original reviewed package missing-artifact reason stays explicit; it is
    not converted into a claim that the package was recovered.
    """
    omissions = []
    for module in closure["modules"]:
        pin = {name: module[name] for name in ("path", "sha256", "kind")}
        raw = store.read(pin)
        if (
            module["git_blob"] is not None
            and git_oid("blob", raw) != module["git_blob"]
        ):
            raise EvidenceError("planned source Git blob differs")
    for artifact in (
        closure["native_backends"]
        + closure["artifacts"]
        + [closure["python"]["executable"]]
    ):
        store.read(artifact)
    for package in closure["packages"]:
        if package["artifact"] is None:
            omissions.append(
                {
                    "package": package["name"],
                    "reason": package["artifact_missing_reason"],
                }
            )
        else:
            store.read(package["artifact"])
    return omissions


def _tree_entries(raw):
    entries = {}
    while raw:
        try:
            head, remaining = raw.split(b"\0", 1)
            mode, name = head.split(b" ", 1)
            oid, raw = remaining[:20], remaining[20:]
            if len(oid) != 20 or name in entries or not name or b"/" in name:
                raise EvidenceError("invalid raw Git tree")
            entries[name] = (mode, oid.hex())
        except ValueError as error:
            raise EvidenceError("invalid raw Git tree encoding") from error
    return entries


def verify_git_catalog_origin(store, proof, catalogs):
    """Raw commit/tree hashes prove actual subtree inclusion, without Git calls."""
    if (
        set(proof) != {"schema", "entries"}
        or proof["schema"] != "gen1-stochastic-git-subtree-proof-v1"
        or len(proof["entries"]) != len(catalogs)
    ):
        raise EvidenceError("exact raw Git origin proof required")
    for entry, catalog in zip(proof["entries"], catalogs, strict=True):
        if set(entry) != {
            "commit",
            "tree",
            "subtree_git_tree",
            "commit_object",
            "root_tree_object",
            "src_tree_object",
        } or any(
            entry[name] != catalog[name]
            for name in ("commit", "tree", "subtree_git_tree")
        ):
            raise EvidenceError("Git origin proof catalog subject differs")
        commit = store.read(entry["commit_object"])
        root = store.read(entry["root_tree_object"])
        src = store.read(entry["src_tree_object"])
        if (
            git_oid("commit", commit) != entry["commit"]
            or commit.split(b"\n", 1)[0] != b"tree " + entry["tree"].encode()
            or git_oid("tree", root) != entry["tree"]
        ):
            raise EvidenceError("catalog commit/root tree not established")
        root_entries, src_entries = _tree_entries(root), _tree_entries(src)
        if root_entries.get(b"src") != (
            b"40000",
            git_oid("tree", src),
        ) or src_entries.get(b"gragra") != (b"40000", catalog["subtree_git_tree"]):
            raise EvidenceError("claimed subtree not actually linked from commit")
