"""Independent stdlib root governor for diagnostic A2 and Cylinder epochs."""

from __future__ import annotations

import base64
import errno
import fcntl
import hashlib
import json
import os
import re
import secrets
import select
import signal
import socket
import stat
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


_RUNTIME_DIR = Path(__file__).resolve(strict=True).parent
sys.path.insert(0, str(_RUNTIME_DIR))
from deadline import DeadlineError, arm_root_deadline, make_pdeath_preexec, parse_proc_timers, verify_new_proc_timer  # noqa: E402


# This is deliberately the first runtime operation: the 900-second kernel
# deadline is derived from the root's kernel starttime before CLI or manifest
# parsing. It remains armed through terminal fsync and os._exit.
try:
    _ROOT_TIMER_EVIDENCE = arm_root_deadline(900, 60)
except BaseException as exc:
    os.write(2, f"diagnostic root deadline refused: {exc}\n".encode("utf-8", "replace"))
    os._exit(127)


from ledger import (  # noqa: E402
    PHASE_SCHEDULE,
    Ledger,
    LedgerError,
    ROOT_MANIFEST_SHA256_PLACEHOLDER,
    bind_root_manifest_sha256,
    validate_authorization,
    validate_cleanup_receipt,
    validate_controller_broker_eof,
    validate_controller_terminal,
    validate_dependency_inventory,
    validate_launch_binding,
    validate_role_inventory,
    verify_chain,
)
from procfs import (  # noqa: E402
    ProcfsError,
    ProcfsGone,
    boot_id,
    cleanup_inventory,
    direct_child_pids,
    enumerate_processes,
    process_identity,
    reap_untracked_waitables,
    root_identity,
    set_child_subreaper,
    task_ids,
)

from source_catalog import (  # noqa: E402
    read_pinned_bytes,
    sha256_bytes,
    sha256_file,
    strict_json,
    verify_interpreter,
    verify_pin,
    verify_source_catalog,
)


class GovernorError(RuntimeError):
    """The epoch cannot continue without violating a frozen root condition."""


class ControllerBrokerEOF(GovernorError):
    """A controller closed its role-request channel."""


_HOST_ADMISSION_DIRECTORY = "/home/washimi/.local/state/gragra/diagnostic-execution/host-worker-admission-v1"
_HOST_ADMISSION_SCHEMA = "gragra.diagnostic.host-worker-admission.v1"
_HOST_ADMISSION_PREPARATION_SCHEMA = "gragra.diagnostic.host-admission-directory-preparation.v1"
_HOST_ADMISSION_CONVENTION = "gragra-host-scientific-worker-cap-2-v1"
_HOST_ADMISSION_SLOTS = ["slot-0", "slot-1"]
_ROLE_KILL_REAP_RESERVE_NS = 250_000_000
_ROLE_POLL_INTERVAL_NS = 50_000_000
_ROOT_TERMINAL_INTENT_CONTRACT = {
    "event": "ROOT_TERMINAL",
    "intent_schema": "gragra-diagnostic-root-terminal-intent-v1",
    "native_exit_receipt": "external_only_required",
}
_LEDGER_OWNED_FIELDS = {
    "event", "run_id", "model", "sequence", "previous_sha256", "event_sha256"
}


class _HostWorkerAdmission:
    """Manifest-bound two-slot flock lease, owned by the charged governor root."""

    def __init__(self, config: Any, *, model: str, epoch_id: str, manifest_sha256: str, root: dict[str, Any]) -> None:
        self.config = config
        self.model = model
        self.epoch_id = epoch_id
        self.manifest_sha256 = manifest_sha256
        self.root = root
        self.fd: int | None = None
        self.slot_id: str | None = None
        self.owner_record: dict[str, Any] | None = None
        self.evidence: dict[str, Any] = {
            "schema": _HOST_ADMISSION_SCHEMA,
            "convention_id": _HOST_ADMISSION_CONVENTION,
            "capacity": 2,
            "status": "not_acquired",
            "slot_id": None,
            "observed_slots": [],
            "release": None,
        }

    @staticmethod
    def _read_fd(fd: int) -> bytes:
        raw = os.pread(fd, 8193, 0)
        if len(raw) > 8192:
            raise GovernorError("host admission owner record exceeds 8 KiB")
        return raw

    def acquire(self) -> dict[str, Any]:
        prep_pin = self.config.get("directory_preparation_pin") if isinstance(self.config, dict) else None
        expected = {
            "schema": _HOST_ADMISSION_SCHEMA,
            "convention_id": _HOST_ADMISSION_CONVENTION,
            "lock_directory": _HOST_ADMISSION_DIRECTORY,
            "capacity": 2,
            "slot_ids": _HOST_ADMISSION_SLOTS,
            "owner_record_schema": "gragra.diagnostic.host-worker-slot-owner.v1",
            "hold_scope": "whole_root_epoch",
            "stale_owner": "refuse_without_reclaim",
            "release": "clear_and_fsync_only_after_confirmed_cleanup",
        }
        self.evidence["directory_preparation"] = {
            "schema": _HOST_ADMISSION_PREPARATION_SCHEMA,
            "step": "Step3",
            "pin": prep_pin,
            "target_directory": _HOST_ADMISSION_DIRECTORY,
            "governor_created_directory": False,
            "directory_exists_before_admission": None,
        }
        if (
            not isinstance(prep_pin, dict)
            or set(prep_pin) != {"path", "sha256"}
            or not isinstance(prep_pin.get("path"), str)
            or not Path(prep_pin["path"]).is_absolute()
            or not isinstance(prep_pin.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", prep_pin["sha256"]) is None
            or {key: value for key, value in self.config.items() if key != "directory_preparation_pin"} != expected
        ):
            self.evidence.update({"status": "refused", "reason": "manifest host admission convention differs"})
            return self.evidence
        try:
            prep_raw = read_pinned_bytes(
                prep_pin["path"], prep_pin["sha256"], label="host admission directory preparation"
            )
            prep = strict_json(prep_raw)
            if prep != {
                "schema": _HOST_ADMISSION_PREPARATION_SCHEMA,
                "step": "Step3",
                "target_directory": _HOST_ADMISSION_DIRECTORY,
                "preparation_action": "prestage_directory_before_root_start",
                "required_owner": "root_effective_uid",
                "required_mode_octal": "0700",
                "must_exist_before_admission": True,
                "governor_may_create_directory": False,
                "slot_files": "governor_may_create_only_after_directory_validation",
            }:
                raise GovernorError("pinned host admission directory preparation contract differs")
        except (OSError, ValueError, GovernorError) as exc:
            self.evidence.update({"status": "refused", "reason": f"host admission preparation pin refused: {exc}"})
            return self.evidence
        directory = Path(_HOST_ADMISSION_DIRECTORY)
        try:
            directory_stat = directory.lstat()
            self.evidence["directory_preparation"]["directory_exists_before_admission"] = True
            if (
                not stat.S_ISDIR(directory_stat.st_mode)
                or directory_stat.st_uid != os.geteuid()
                or stat.S_IMODE(directory_stat.st_mode) != 0o700
            ):
                raise GovernorError("host admission directory identity or mode is unsafe")
            for slot_id in _HOST_ADMISSION_SLOTS:
                path = directory / f"{slot_id}.lock"
                flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
                fd = os.open(path, flags, 0o600)
                info = os.fstat(fd)
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_uid != os.geteuid()
                    or stat.S_IMODE(info.st_mode) != 0o600
                    or info.st_nlink != 1
                ):
                    os.close(fd)
                    raise GovernorError(f"host admission slot file is unsafe: {slot_id}")
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    owner_raw = _HostWorkerAdmission._read_fd(fd)
                    try:
                        owner_record = strict_json(owner_raw) if owner_raw else None
                        owner_parse_error = None
                    except Exception as exc:
                        owner_record = None
                        owner_parse_error = f"{type(exc).__name__}: {exc}"
                    self.evidence["observed_slots"].append(
                        {
                            "slot_id": slot_id,
                            "state": "locked_by_convention_owner",
                            "owner_bytes_sha256": hashlib.sha256(owner_raw).hexdigest(),
                            "owner_record": owner_record,
                            "owner_parse_error": owner_parse_error,
                            "grant_from_locked_slot": False,
                        }
                    )
                    os.close(fd)
                    continue
                self.fd = fd
                existing = _HostWorkerAdmission._read_fd(fd)
                if existing:
                    try:
                        stale_owner = strict_json(existing)
                        stale_owner_error = None
                    except Exception as exc:
                        stale_owner = None
                        stale_owner_error = f"{type(exc).__name__}: {exc}"
                    self.evidence["observed_slots"].append(
                        {
                            "slot_id": slot_id,
                            "state": "stale_owner_refused",
                            "owner_bytes_sha256": hashlib.sha256(existing).hexdigest(),
                            "owner_record": stale_owner,
                            "owner_parse_error": stale_owner_error,
                            "grant_from_stale_slot": False,
                        }
                    )
                    fcntl.flock(fd, fcntl.LOCK_UN)
                    os.close(fd)
                    self.fd = None
                    self.evidence.update({"status": "refused", "reason": f"unlocked slot contains stale owner evidence: {slot_id}"})
                    return self.evidence
                owner = {
                    "schema": "gragra.diagnostic.host-worker-slot-owner.v1",
                    "convention_id": _HOST_ADMISSION_CONVENTION,
                    "capacity": 2,
                    "slot_id": slot_id,
                    "model": self.model,
                    "epoch_id": self.epoch_id,
                    "manifest_sha256": self.manifest_sha256,
                    "pid": self.root["pid"],
                    "starttime_ticks": self.root["starttime_ticks"],
                    "boot_id": self.root["boot_id"],
                    "acquired_monotonic_ns": time.monotonic_ns(),
                }
                raw = _canonical(owner)
                os.ftruncate(fd, 0)
                offset = 0
                while offset < len(raw):
                    written = os.pwrite(fd, raw[offset:], offset)
                    if written <= 0:
                        raise GovernorError("host admission owner record short write")
                    offset += written
                os.fsync(fd)
                self.slot_id = slot_id
                self.owner_record = owner
                self.evidence.update(
                    {
                        "status": "acquired",
                        "slot_id": slot_id,
                        "owner": owner,
                        "owner_record_sha256": hashlib.sha256(raw).hexdigest(),
                        "lock_path": str(path),
                        "observed_slots": [*self.evidence["observed_slots"], {"slot_id": slot_id, "state": "acquired_empty_slot"}],
                    }
                )
                return self.evidence
        except (OSError, GovernorError) as exc:
            if isinstance(exc, FileNotFoundError):
                self.evidence["directory_preparation"]["directory_exists_before_admission"] = False
            if self.fd is not None:
                try:
                    fcntl.flock(self.fd, fcntl.LOCK_UN)
                    os.close(self.fd)
                except OSError:
                    pass
                self.fd = None
            self.evidence.update({"status": "refused", "reason": f"{type(exc).__name__}: {exc}"})
            return self.evidence
        self.evidence.update({"status": "refused", "reason": "both manifest-bound host admission slots are locked"})
        return self.evidence

    @staticmethod
    def _sync_directory(path: Path) -> None:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def release(self, *, cleanup_confirmed: bool) -> dict[str, Any]:
        if self.fd is None:
            release = {"status": "not_owned", "cleanup_confirmed": cleanup_confirmed}
            self.evidence["release"] = release
            return release
        release: dict[str, Any] = {
            "slot_id": self.slot_id,
            "lock_path": self.evidence["lock_path"],
            "cleanup_confirmed": cleanup_confirmed,
            "owner_record_sha256": self.evidence["owner_record_sha256"],
        }
        try:
            if cleanup_confirmed:
                os.ftruncate(self.fd, 0)
                os.fsync(self.fd)
                _HostWorkerAdmission._sync_directory(Path(_HOST_ADMISSION_DIRECTORY))
                release["status"] = "released_after_confirmed_cleanup"
                release["owner_record_cleared"] = True
            else:
                release["status"] = "stale_owner_retained_cleanup_unconfirmed"
                release["owner_record_cleared"] = False
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None
        except OSError as exc:
            release.update({"status": "release_unconfirmed", "owner_record_cleared": False, "error": f"{type(exc).__name__}: {exc}"})
        self.evidence["release"] = release
        return release


class _CgroupMemoryLimit:
    """Fresh cgroup-v2 hard limit for root plus inherited direct children."""

    LIMIT_BYTES = 4 * 1024 * 1024 * 1024

    def __init__(self, root_pid: int, start_ticks: int, parent: Path, target: Path, relative: str) -> None:
        self.root_pid = root_pid
        self.parent = parent
        self.target = target
        self.relative = relative
        self.initial_events = self._events()
        self.peak_bytes = self._integer("memory.current")
        self.closed = False
        self.setup_evidence = {
            "policy": "hard_cgroup_v2_memory.max",
            "mount": str(Path("/sys/fs/cgroup").resolve(strict=True)),
            "parent": str(parent),
            "target": str(target),
            "relative_target": relative,
            "limit_bytes": self.LIMIT_BYTES,
            "memory_max_readback": self._integer("memory.max"),
            "initial_memory_events": self.initial_events,
        }

    @classmethod
    def try_enable(cls, root_pid: int, start_ticks: int) -> "_CgroupMemoryLimit | None":
        mount = Path("/sys/fs/cgroup")
        target: Path | None = None
        created = False
        parent: Path | None = None
        root_moved = False
        try:
            if not (mount / "cgroup.controllers").is_file():
                return None
            rows = [line.split(":", 2) for line in Path("/proc/self/cgroup").read_text().splitlines()]
            unified = [row[2] for row in rows if len(row) == 3 and row[0] == "0" and row[1] == ""]
            if len(unified) != 1 or not unified[0].startswith("/"):
                return None
            relative_parent = Path(unified[0].lstrip("/"))
            if any(part in {"", ".", ".."} for part in relative_parent.parts):
                return None
            mount_real = mount.resolve(strict=True)
            parent = mount_real.joinpath(*relative_parent.parts).resolve(strict=True)
            parent.relative_to(mount_real)
            controllers = set((parent / "cgroup.controllers").read_text().split())
            if "memory" not in controllers:
                return None
            target = parent / f"gragra-diagnostic-{root_pid}-{start_ticks}"
            target.mkdir(mode=0o700)
            created = True
            (target / "memory.max").write_text(f"{cls.LIMIT_BYTES}\n")
            if (target / "memory.max").read_text().strip() != str(cls.LIMIT_BYTES):
                raise GovernorError("cgroup-v2 memory.max did not read back the fixed 4 GiB limit")
            relative = str(target.relative_to(mount_real))
            (target / "cgroup.procs").write_text(f"{root_pid}\n")
            root_moved = True
            current = [line.split(":", 2)[2] for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::")]
            if current != ["/" + relative]:
                raise GovernorError("root did not move into its fresh hard-limit cgroup")
            return cls(root_pid, start_ticks, parent, target, relative)
        except BaseException as exc:
            cleanup_errors = []
            if root_moved and parent is not None:
                try:
                    (parent / "cgroup.procs").write_text(f"{root_pid}\n")
                    mount_real = mount.resolve(strict=True)
                    current = [
                        line.split(":", 2)[2]
                        for line in Path("/proc/self/cgroup").read_text().splitlines()
                        if line.startswith("0::")
                    ]
                    if current != ["/" + str(parent.relative_to(mount_real))]:
                        cleanup_errors.append("root cgroup membership did not read back at parent")
                except (OSError, ValueError) as cleanup_exc:
                    cleanup_errors.append(f"root could not be moved back: {cleanup_exc}")
            if created and target is not None:
                try:
                    target.rmdir()
                except OSError as cleanup_exc:
                    cleanup_errors.append(f"partial cgroup directory could not be removed: {cleanup_exc}")
            if cleanup_errors:
                raise GovernorError(
                    "partial cgroup setup cleanup is unconfirmed: " + "; ".join(cleanup_errors)
                ) from exc
            if isinstance(exc, OSError) and exc.errno in {
                errno.EACCES,
                errno.EPERM,
                errno.EROFS,
                errno.ENOENT,
                errno.ENODEV,
                errno.EOPNOTSUPP,
            }:
                return None
            if isinstance(exc, GovernorError):
                raise
            raise GovernorError(f"cgroup-v2 hard limit setup failed: errno={exc.errno}") from exc

    def _integer(self, name: str) -> int:
        raw = (self.target / name).read_text().strip()
        if not raw.isdecimal():
            raise GovernorError(f"cgroup {name} readback is malformed")
        return int(raw)

    def _events(self) -> dict[str, int]:
        values: dict[str, int] = {}
        for line in (self.target / "memory.events").read_text().splitlines():
            parts = line.split()
            if len(parts) != 2 or not parts[1].isdecimal() or parts[0] in values:
                raise GovernorError("cgroup memory.events record is malformed")
            values[parts[0]] = int(parts[1])
        if not {"oom", "oom_kill"}.issubset(values):
            raise GovernorError("cgroup memory.events lacks OOM counters")
        return values

    def sample(self) -> dict[str, Any]:
        current = self._integer("memory.current")
        events = self._events()
        self.peak_bytes = max(self.peak_bytes, current)
        if events["oom"] > self.initial_events["oom"] or events["oom_kill"] > self.initial_events["oom_kill"]:
            raise GovernorError("cgroup hard memory limit triggered an OOM event")
        if current > self.LIMIT_BYTES:
            raise GovernorError("cgroup memory.current exceeds the fixed 4 GiB hard limit")
        return {"current_bytes": current, "peak_bytes": self.peak_bytes, "memory_events": events}

    def close(self) -> dict[str, Any]:
        if self.closed:
            raise GovernorError("cgroup cleanup was requested more than once")
        members_raw = (self.target / "cgroup.procs").read_text().split()
        if any(not value.isdecimal() or int(value) <= 0 for value in members_raw):
            raise GovernorError("cgroup.procs contains a malformed member PID")
        members = [int(value) for value in members_raw]
        no_unknown_members = set(members) <= {self.root_pid}
        (self.parent / "cgroup.procs").write_text(f"{self.root_pid}\n")
        current = [line.split(":", 2)[2] for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::")]
        moved_back = current == ["/" + str(self.parent.relative_to(Path("/sys/fs/cgroup").resolve(strict=True)))]
        try:
            self.target.rmdir()
            removed = True
        except OSError:
            removed = False
        self.closed = True
        return {
            "policy": "hard_cgroup_v2_memory.max",
            "limit_bytes": self.LIMIT_BYTES,
            "peak_bytes": self.peak_bytes,
            "members_before_root_move": members,
            "no_unknown_members": no_unknown_members,
            "root_moved_back": moved_back,
            "cgroup_removed": removed,
            "confirmed": no_unknown_members and moved_back and removed,
        }


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as exc:
        raise GovernorError("evidence object is not finite canonical JSON") from exc


def _read_bytes(path: Path, *, limit: int = 64 * 1024 * 1024) -> bytes:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise GovernorError(f"pinned input cannot be opened without following a symlink: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise GovernorError(f"pinned input is not a regular file: {path}")
        chunks = []
        size = 0
        while True:
            block = os.read(descriptor, min(1024 * 1024, limit + 1 - size))
            if not block:
                break
            chunks.append(block)
            size += len(block)
            if size > limit:
                raise GovernorError(f"pinned input exceeds its bounded size: {path}")
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after or size != after.st_size:
            raise GovernorError(f"pinned input changed while its bytes were captured: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _write_new(path: Path, raw: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        view = memoryview(raw)
        while view:
            count = os.write(fd, view)
            if count <= 0:
                raise GovernorError(f"short evidence write: {path}")
            view = view[count:]
        os.fsync(fd)
    finally:
        os.close(fd)
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _sync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _normalize_source_catalog(raw: Any, *, manifest: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize the original A2 map or C file list without replacing its manifest."""
    candidate = manifest.get("candidate")
    configured_roots = (
        candidate.get("source_scan_roots") if isinstance(candidate, dict) else None
    )
    if raw is None and isinstance(candidate, dict):
        raw = candidate.get("source_catalog")
    if not isinstance(raw, (dict, list)):
        raise GovernorError("source catalog is not an A2 map or Cylinder list")
    if isinstance(raw, dict) and set(raw) == {"path", "sha256"}:
        raise GovernorError("pinned source catalog must be decoded before normalization")

    supplied_inventory = None
    if isinstance(raw, list):
        rows = raw
    else:
        if not set(raw).issubset({"files", "inventory", "scan_roots"}) or "files" not in raw:
            raise GovernorError("source catalog fields differ from the canonical adapter")
        configured_roots = raw.get("scan_roots", configured_roots)
        supplied_inventory = raw.get("inventory")
        values = raw.get("files")
        if isinstance(values, list):
            rows = values
        elif isinstance(values, dict):
            rows = [{"path": name, "sha256": digest} for name, digest in values.items()]
        else:
            raise GovernorError("source catalog files must be a map or original C row list")

    files: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise GovernorError("source catalog row is malformed")
        name, digest = row["path"], row["sha256"]
        if (
            not isinstance(name, str)
            or not name
            or name.startswith("/")
            or "\\" in name
            or any(part in {"", ".", ".."} for part in name.split("/"))
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or name in files
        ):
            raise GovernorError("source catalog path/hash is malformed or duplicated")
        files[name] = digest
    inventory = sorted(files)
    if not inventory:
        raise GovernorError("source catalog is empty")
    if supplied_inventory is not None and supplied_inventory != inventory:
        raise GovernorError("source catalog inventory contradicts its file pins")
    if configured_roots is None:
        if any("/" not in name for name in inventory):
            scan_roots = ["."]
        else:
            # Keep the entire top-level project source tree in the completeness
            # scan so an unlisted importable sibling cannot escape the catalog.
            scan_roots = sorted({name.split("/", 1)[0] for name in inventory})
    else:
        scan_roots = configured_roots
    if (
        not isinstance(scan_roots, list)
        or not scan_roots
        or any(
            not isinstance(path, str)
            or path == ""
            or path.startswith("/")
            or "\\" in path
            or (path != "." and any(part in {"", ".", ".."} for part in path.split("/")))
            for path in scan_roots
        )
        or scan_roots != sorted(set(scan_roots))
    ):
        raise GovernorError("source catalog scan roots are malformed or nondeterministic")
    return {"files": files, "inventory": inventory, "scan_roots": scan_roots}


def _manifest_source_catalog(manifest: dict[str, Any]) -> dict[str, Any]:
    raw = manifest.get("source_catalog")
    if raw is None and isinstance(manifest.get("candidate"), dict):
        raw = manifest["candidate"].get("source_catalog")
    if isinstance(raw, dict) and set(raw) == {"path", "sha256"}:
        if not isinstance(raw.get("path"), str) or not isinstance(raw.get("sha256"), str):
            raise GovernorError("source catalog pin is malformed")
        captured = read_pinned_bytes(raw["path"], raw["sha256"], label="source catalog")
        decoded = strict_json(captured)
        return _normalize_source_catalog(decoded, manifest=manifest)
    return _normalize_source_catalog(raw, manifest=manifest)


def _all_roles(phase: dict[str, Any]) -> list[dict[str, Any]]:
    roles = phase["roles"]
    if isinstance(roles, dict):
        return [{"role": name, "max_seconds": cap} for name, cap in roles.items()]
    return [{"role": item["role"], "max_seconds": item["max_seconds"]} for item in roles]


def _header(manifest: dict[str, Any], model: str) -> dict[str, Any]:
    if model == "A2":
        return manifest
    if manifest.get("schema") != "gragra.cylinder.diagnostic.manifest.v1":
        raise GovernorError("Cylinder family manifest schema differs")
    header = manifest.get("runtime_authorization")
    if not isinstance(header, dict):
        raise GovernorError("Cylinder runtime_authorization header is missing")
    return header


def _safe_manifest_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise GovernorError(f"{label} must be an absolute path")
    path = Path(value)
    if path.is_symlink():
        raise GovernorError(f"{label} cannot be a symlink")
    return path


def _parse_control_line(sock: socket.socket, deadline_ns: int) -> dict[str, Any] | None:
    data = bytearray()
    while len(data) <= 1_048_576:
        remaining = (deadline_ns - time.monotonic_ns()) / 1_000_000_000
        if remaining <= 0:
            return None
        ready, _, _ = select.select([sock], [], [], min(remaining, 0.25))
        if not ready:
            continue
        chunk = sock.recv(1)
        if not chunk:
            raise GovernorError("child control socket closed mid-record")
        if chunk == b"\n":
            value = strict_json(bytes(data))
            if not isinstance(value, dict):
                raise GovernorError("child control record is not a JSON object")
            return value
        data.extend(chunk)
    raise GovernorError("child control record exceeds fixed size")


def _receive_controller_record(handle: dict[str, Any], deadline_ns: int) -> dict[str, Any] | None:
    """Read one bounded line without blocking the root's deadline/RSS loop."""
    sock = handle["socket"]
    buffered: bytearray = handle["control_buffer"]
    while True:
        newline = buffered.find(b"\n")
        if newline >= 0:
            raw = bytes(buffered[:newline])
            del buffered[: newline + 1]
            value = strict_json(raw)
            if not isinstance(value, dict):
                raise GovernorError("controller request is not a JSON object")
            return value
        if len(buffered) > 1_048_576:
            raise GovernorError("controller request exceeds the fixed control limit")
        remaining = deadline_ns - time.monotonic_ns()
        if remaining <= 0:
            return None
        ready, _, _ = select.select([sock], [], [], min(remaining / 1_000_000_000, 0.25))
        if not ready:
            return None
        try:
            chunk = sock.recv(65536)
        except BlockingIOError:
            continue
        if not chunk:
            if buffered:
                raise GovernorError("controller closed broker channel mid-record")
            raise ControllerBrokerEOF("controller closed its broker channel")
        buffered.extend(chunk)


def _send_control(sock: socket.socket, value: dict[str, Any]) -> None:
    raw = _canonical(value)
    if len(raw) > 1_048_576:
        raise GovernorError("root control record exceeds fixed size")
    sock.sendall(raw)


def _proc_timer_raw(pid: int) -> bytes:
    try:
        return Path(f"/proc/{pid}/timers").read_bytes()
    except OSError as exc:
        raise GovernorError(f"child POSIX timer inventory unavailable: errno={exc.errno}") from exc


class RootGovernor:
    def __init__(self, model: str, manifest_path: Path, root_timer: dict[str, Any]) -> None:
        self.model = model
        self.manifest_path = manifest_path
        self.root_timer = root_timer
        self.started_mono_ns = time.monotonic_ns()
        self.hard_deadline_ns = root_timer["hard_deadline_mono_ns"]
        self.soft_deadline_ns = root_timer["soft_deadline_mono_ns"]
        self.root = root_identity()
        self.root_stat_identity = {
            key: self.root[key]
            for key in ("pid", "ppid", "pgrp", "session", "starttime_ticks")
        }
        self.root_boot_id = self.root["boot_id"]
        self.subreaper = set_child_subreaper()
        self.manifest_raw = _read_bytes(manifest_path)
        self.manifest_sha256 = hashlib.sha256(self.manifest_raw).hexdigest()
        value = strict_json(self.manifest_raw)
        if not isinstance(value, dict):
            raise GovernorError("external manifest must be a JSON object")
        self.manifest = value
        self.auth_header = _header(value, model)
        self.auth = validate_authorization(self.auth_header, expected_model=model)
        if self.auth_header.get("terminal_intent_contract") != _ROOT_TERMINAL_INTENT_CONTRACT:
            raise GovernorError("manifest root terminal-intent event/schema contract differs")
        self.epoch_id = value.get("epoch_id")
        if not isinstance(self.epoch_id, str) or not self.epoch_id or len(self.epoch_id) > 128:
            raise GovernorError("epoch_id must be a fresh bounded string")
        self.ledger_id = f"{model}:{self.epoch_id}"
        paths = value.get("paths")
        if not isinstance(paths, dict):
            raise GovernorError("manifest paths object is missing")
        if self.model == "A2":
            try:
                a2_runtime = _RUNTIME_DIR.parents[1] / "stochastic_formal" / "a2" / "v2"
                if not a2_runtime.is_dir():
                    raise GovernorError("A2 manifest adapter is absent from its candidate worktree")
                sys.path.insert(0, str(a2_runtime))
                from manifest import validate_retained_reference_object

                validate_retained_reference_object(value)
            except (ImportError, ValueError) as exc:
                raise GovernorError(f"A2 retained-reference manifest object is invalid: {exc}") from exc
        self.worktree_root = _safe_manifest_path(paths.get("worktree_root"), "worktree_root").resolve(strict=True)
        self.output_root = _safe_manifest_path(paths.get("output_root"), "output_root")
        if not self.worktree_root.is_dir() or self.output_root.exists():
            raise GovernorError("worktree is absent or output_root already exists")
        if self.output_root == self.worktree_root or self.worktree_root in self.output_root.parents:
            raise GovernorError("external output root must be outside the candidate worktree")
        self.output_root.mkdir(parents=True, mode=0o700)
        self._fsync_parent(self.output_root.parent)
        self.ledger = Ledger(self.output_root / "ledger.jsonl", run_id=self.epoch_id, model=model)
        python = value.get("python")
        if not isinstance(python, dict):
            raise GovernorError("Python pin object is missing")
        self.python_evidence = verify_interpreter(python)
        self.python_path = self.python_evidence["executable"]
        self.site_packages = python.get("site_packages")
        if not isinstance(self.site_packages, list):
            raise GovernorError("site-packages inventory is missing")
        for item in self.site_packages:
            path = _safe_manifest_path(item, "site-packages path").resolve(strict=True)
            if not path.is_dir():
                raise GovernorError("site-packages pin is not a directory")
        shared = value.get("shared_runtime")
        if not isinstance(shared, dict):
            raise GovernorError("shared_runtime pins are missing")
        self.shared_pins = {
            name: verify_pin(shared[name], label=f"shared runtime {name}")
            for name in ("archive", "bootstrap", "governor")
        }
        if self.shared_pins["governor"]["path"] != str(Path(__file__).resolve(strict=True)):
            raise GovernorError("manifest governor path is not the running source")
        self.catalog = _manifest_source_catalog(value)
        self.source_evidence = verify_source_catalog(self.worktree_root, self.catalog)
        self._verify_external_pins(value.get("external_pins"))
        self.dependency_inventory = []
        if self.model == "A2":
            try:
                self.dependency_inventory = validate_dependency_inventory(
                    value.get("dependency_inventory")
                )
            except LedgerError as exc:
                raise GovernorError(str(exc)) from exc
            external_dependencies = {
                (pin["path"], pin["sha256"]) for pin in self.external_pin_evidence.values()
            }
            for pin in self.dependency_inventory:
                resolved_path = str(Path(pin["path"]).resolve(strict=True))
                if (resolved_path, pin["sha256"]) not in external_dependencies:
                    raise GovernorError(
                        f"A2 dependency pin is absent from immutable external_pins: {pin['module']}"
                    )
        self.jobs = self._validate_jobs(value.get("runtime_jobs"))
        self.environment = self._validate_environment(value.get("environment"))
        self.children: dict[str, dict[str, Any]] = {}
        self.reaped: list[dict[str, Any]] = []
        self.declared: list[dict[str, Any]] = []
        self.all_handles: dict[str, dict[str, Any]] = {}
        self.unbound_children: list[dict[str, Any]] = []
        self.unbound_child_evidence: list[dict[str, Any]] = []
        self.cgroup = _CgroupMemoryLimit.try_enable(
            self.root["pid"], self.root["starttime_ticks"]
        )
        self.rss_policy = (
            "hard_cgroup_v2_memory.max" if self.cgroup is not None else "soft_1s_procfs"
        )
        self.cgroup_sample: dict[str, Any] | None = None
        self.rss_peak_bytes = self.cgroup.peak_bytes if self.cgroup is not None else 0
        self.rss_peak_mono_ns = time.monotonic_ns()
        self.rss_sample_count = 0
        self.rss_next_sample_ns = 0
        self.active_phase: str | None = None
        self.phase_start_ns = 0
        self.phase_deadline_ns = 0
        self.phase_roles: dict[str, dict[str, Any]] = {}
        self.phase_controller_role: str | None = None
        self.phase_dispositions: dict[str, str] = {}
        self.phase_started: dict[str, dict[str, Any]] = {}
        self.role_maximum_events: list[dict[str, Any]] = []
        self.timing_reconciliation: dict[str, Any] | None = None
        self.phase_reservations: list[dict[str, Any]] = []
        self.failed_reason: str | None = None
        self.cleanup_started = False
        self.host_admission = _HostWorkerAdmission(
            self.auth_header.get("host_worker_admission"),
            model=self.model,
            epoch_id=self.epoch_id,
            manifest_sha256=self.manifest_sha256,
            root=self.root,
        )
        self.host_admission_evidence = self.host_admission.acquire()
        self._root_start_record()

    @staticmethod
    def _fsync_parent(path: Path) -> None:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _verify_external_pins(self, pins: Any) -> None:
        if not isinstance(pins, dict):
            raise GovernorError("external_pins must be a named mapping")
        if not pins:
            raise GovernorError("external_pins cannot be empty")
        observed = {}
        for name, pin in pins.items():
            if not isinstance(name, str) or not name:
                raise GovernorError("external pin name is malformed")
            observed[name] = verify_pin(pin, label=f"external {name}")
        self.external_pin_evidence = observed

    def _validate_environment(self, raw: Any) -> dict[str, str]:
        if not isinstance(raw, dict) or set(raw) != {"allowlist", "values"}:
            raise GovernorError("worker environment inventory differs from the fixed schema")
        allowlist, values = raw["allowlist"], raw["values"]
        dynamic = {
            "GRAGRA_DIAGNOSTIC_CONTROL_FD",
            "GRAGRA_DIAGNOSTIC_LAUNCH_TOKEN",
            "GRAGRA_DIAGNOSTIC_MODEL",
            "GRAGRA_DIAGNOSTIC_PHASE",
            "GRAGRA_DIAGNOSTIC_LEDGER_ID",
        }
        if (
            not isinstance(allowlist, list)
            or allowlist != sorted(allowlist)
            or len(allowlist) != len(set(allowlist))
            or not isinstance(values, dict)
            or set(values) != set(allowlist) - dynamic
            or not dynamic.issubset(set(allowlist))
            or any(not isinstance(key, str) or not isinstance(value, str) for key, value in values.items())
        ):
            raise GovernorError("worker environment allowlist/values contradict")
        forbidden = {"PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE", "GIT_DIR", "GIT_WORK_TREE"}
        if forbidden & set(allowlist):
            raise GovernorError("worker environment contains an ambient path override")
        required = {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
        }
        if any(values.get(name) != value for name, value in required.items()):
            raise GovernorError("numerical worker thread environment is not fixed to one")
        return dict(values)

    def _validate_jobs(self, raw: Any) -> dict[tuple[str, str], dict[str, Any]]:
        if not isinstance(raw, list):
            raise GovernorError("runtime_jobs must be a frozen list")
        expected: dict[tuple[str, str], dict[str, Any]] = {}
        phase_controllers: dict[str, str | None] = {}
        for phase in PHASE_SCHEDULE[self.model]:
            for role in _all_roles(phase):
                expected[(phase["phase_id"], role["role"])] = role
            declared_roles = phase.get("roles")
            controllers = (
                [item["role"] for item in declared_roles if item.get("kind") == "controller"]
                if isinstance(declared_roles, list)
                else []
            )
            if len(controllers) > 1:
                raise GovernorError("phase schedule declares multiple root-broker controllers")
            phase_controllers[phase["phase_id"]] = controllers[0] if controllers else None
        if len(raw) != len(expected):
            raise GovernorError("runtime_jobs must inventory every reserved role exactly once")
        observed: dict[tuple[str, str], dict[str, Any]] = {}
        required_fields = {
            "phase_id",
            "role",
            "entry_path",
            "entry_sha256",
            "expected_terminal",
            "logical_controller_role",
            "result_path",
            "stdin_payload",
        }
        for item in raw:
            if not isinstance(item, dict) or set(item) != required_fields:
                raise GovernorError("runtime job fields differ from the stable interface")
            key = (item.get("phase_id"), item.get("role"))
            if key not in expected or key in observed:
                raise GovernorError("runtime job is unreserved or duplicated")
            if self.model == "A2":
                controller = phase_controllers[key[0]]
                required_controller = None if key[1] == controller else controller
                if item.get("logical_controller_role") != required_controller:
                    raise GovernorError("A2 logical controller mapping differs from the frozen phase")
            entry, digest = item["entry_path"], item["entry_sha256"]
            if entry is None:
                if digest is not None or item["stdin_payload"] is not None:
                    raise GovernorError("unused role has entry bytes or stdin payload")
            else:
                entry_path = _safe_manifest_path(entry, "runtime job entry").resolve(strict=True)
                if not isinstance(digest, str) or hashlib.sha256(_read_bytes(entry_path)).hexdigest() != digest:
                    raise GovernorError("runtime job entry differs from its SHA256 pin")
                if not isinstance(item["stdin_payload"], dict):
                    raise GovernorError("active runtime job lacks its single JSON payload")
                if self.model == "A2":
                    try:
                        relative_entry = entry_path.relative_to(self.worktree_root).as_posix()
                    except ValueError as exc:
                        raise GovernorError("A2 job entry is outside the pinned candidate worktree") from exc
                    if self.catalog["files"].get(relative_entry) != digest:
                        raise GovernorError("A2 job entry is absent from the exact source catalog")
                    expected_entry_name = "runner.py" if key[1] == controller else "worker.py"
                    if entry_path.name != expected_entry_name:
                        raise GovernorError("A2 role entry differs from the fixed runner/worker source")
                    payload = item["stdin_payload"]
                    expected_payload_fields = {
                        "schema",
                        "model",
                        "phase_id",
                        "role",
                        "manifest_path",
                        "manifest_sha256",
                        "output_root",
                        "result_path",
                    }
                    if (
                        set(payload) != expected_payload_fields
                        or payload.get("schema") != "gragra.a2.diagnostic.worker-request.v1"
                        or payload.get("model") != "A2"
                        or payload.get("phase_id") != key[0]
                        or payload.get("role") != key[1]
                        or payload.get("manifest_path") != str(self.manifest_path)
                        or payload.get("manifest_sha256") != ROOT_MANIFEST_SHA256_PLACEHOLDER
                        or payload.get("output_root") != str(self.output_root)
                        or payload.get("result_path") != item["result_path"]
                    ):
                        raise GovernorError("A2 worker request differs from its root-bound launch job")
            result = _safe_manifest_path(item["result_path"], "runtime job result")
            try:
                result.resolve(strict=False).relative_to(self.output_root.resolve(strict=False))
            except ValueError as exc:
                raise GovernorError("runtime job result path escapes output_root") from exc
            if self.model == "A2":
                expected_result = (
                    self.output_root
                    / "phases"
                    / key[0]
                    / "roles"
                    / key[1]
                    / "result.json"
                )
                if result != expected_result:
                    raise GovernorError("A2 role result path differs from its frozen phase/role location")
            controller = item["logical_controller_role"]
            if controller is not None and not isinstance(controller, str):
                raise GovernorError("logical controller role is malformed")
            terminal = item["expected_terminal"]
            timer_role = (
                self.model == "A2"
                and key == ("accounting", "accounting_timer_fixture")
            )
            timer_profile = {
                "profile": "a2-accounting-child-timer-sigkill-v1",
                "model": "A2",
                "phase": "accounting",
                "role": "accounting_timer_fixture",
                "signal": 9,
                "source": "child_owned_posix_timer",
            }
            if timer_role:
                if terminal != timer_profile or entry is None:
                    raise GovernorError("the contained timer fixture requires its exact active expected-negative profile")
            elif terminal not in (None, "success"):
                raise GovernorError("expected-terminal profiles are forbidden outside the contained timer role")
            observed[key] = item
        if set(observed) != set(expected):
            raise GovernorError("runtime job role inventory is incomplete")
        if self.model == "A2":
            expected_active = {
                ("environment", "environment_leaf"),
                ("accounting", "accounting_test_controller"),
                ("accounting", "accounting_timer_fixture"),
                ("accounting", "accounting_pdeath_refusal"),
                ("accounting", "accounting_handshake_leaf"),
                ("accounting", "accounting_cleanup_leaf"),
                ("retained", "retained_leaf"),
                ("metadata", "metadata_controller"),
                ("metadata", "metadata_worker"),
                ("timing", "timing_controller"),
                ("timing", "timing_worker"),
                ("terminal", "record_leaf"),
            }
            observed_active = {key for key, item in observed.items() if item["entry_path"] is not None}
            if observed_active != expected_active:
                raise GovernorError("A2 active runtime job set differs from the fixed role profile")
        for (phase_id, role), item in observed.items():
            logical = item["logical_controller_role"]
            if logical is not None and (phase_id, logical) not in expected:
                raise GovernorError("runtime job names an unreserved logical controller")
        return observed

    def _root_start_record(self) -> None:
        record = {
            "schema": "gragra-diagnostic-root-start-v1",
            "model": self.model,
            "epoch_id": self.epoch_id,
            "ledger_id": self.ledger_id,
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "root_identity": self.root,
            "timer": self.root_timer,
            "subreaper": self.subreaper,
            "python": self.python_evidence,
            "shared_runtime": self.shared_pins,
            "source_catalog": self.source_evidence,
            "external_pins": self.external_pin_evidence,
            "host_worker_admission": self.host_admission_evidence,
            "rss_policy": self.rss_policy,
            "cgroup_memory_limit": (
                self.cgroup.setup_evidence
                if self.cgroup is not None
                else {
                    "policy": "soft_1s_procfs",
                    "limit_bytes": 4 * 1024 * 1024 * 1024,
                    "sample_interval_seconds": 1,
                    "possible_overshoot": True,
                }
            ),
        }
        _write_new(self.output_root / "root-start.json", _canonical(record))
        self.ledger.append(
            "ROOT_START",
            reserved_seconds=900,
            pid=self.root["pid"],
            starttime_ticks=self.root["starttime_ticks"],
            boot_id=self.root_boot_id,
            manifest_sha256=self.manifest_sha256,
            root_deadline_monotonic_ns=self.hard_deadline_ns,
            soft_deadline_monotonic_ns=self.soft_deadline_ns,
            timer_record_sha256=sha256_bytes(_canonical(self.root_timer)),
            source_catalog_sha256=self.source_evidence["catalog_sha256"],
            subreaper_readback=self.subreaper["readback"],
            external_pins={
                "host_admission_preparation": self.external_pin_evidence[
                    "host_admission_preparation"
                ]
            },
            host_worker_admission=self.host_admission.evidence,
        )

    def _verify_source_binding(self) -> None:
        current = verify_source_catalog(self.worktree_root, self.catalog)
        if current["catalog_sha256"] != self.source_evidence["catalog_sha256"]:
            raise GovernorError("candidate source catalog changed during the root epoch")
        for pin_name, pin in self.shared_pins.items():
            if sha256_file(pin["path"]) != pin["sha256"]:
                raise GovernorError(f"shared runtime source changed during epoch: {pin_name}")

    def _verify_test_framework_sources(self) -> None:
        """Recheck pytest and its source tree immediately before its charged runner."""
        pins = self.manifest.get("external_pins")
        pin = pins.get("pytest_framework_sources") if isinstance(pins, dict) else None
        if not isinstance(pin, dict) or set(pin) != {"path", "sha256"}:
            raise GovernorError("pytest framework source inventory pin is absent")
        raw = read_pinned_bytes(pin["path"], pin["sha256"], label="pytest framework source inventory")
        inventory = strict_json(raw)
        if (
            not isinstance(inventory, dict)
            or inventory.get("schema") != "gragra.a2.test-framework-source-inventory.v1"
            or not isinstance(inventory.get("package_root"), str)
            or not isinstance(inventory.get("distributions"), list)
            or not isinstance(inventory.get("files"), list)
        ):
            raise GovernorError("pytest framework source inventory is malformed before accounting")
        package_root = Path(inventory["package_root"]).resolve(strict=True)
        expected = set()
        for package in ("pytest", "_pytest", "pluggy", "packaging", "iniconfig", "pygments"):
            directory = package_root / package
            if directory.is_symlink() or not directory.is_dir():
                raise GovernorError(f"pytest framework directory is absent or a symlink: {package}")
            expected.update(str(path) for path in directory.rglob("*.py") if path.is_file())
        for distribution in inventory["distributions"]:
            if not isinstance(distribution, dict) or not isinstance(distribution.get("metadata_path"), str):
                raise GovernorError("pytest distribution metadata path is malformed")
            expected.add(distribution["metadata_path"])
        files = inventory["files"]
        if any(not isinstance(item, dict) or set(item) != {"path", "sha256"} for item in files):
            raise GovernorError("pytest source file pin rows are malformed")
        observed = {item["path"]: item["sha256"] for item in files}
        if len(observed) != len(files) or set(observed) != expected:
            raise GovernorError("pytest framework source inventory has unlisted or missing files")
        for path, expected_sha256 in observed.items():
            if not isinstance(expected_sha256, str) or sha256_file(path) != expected_sha256:
                raise GovernorError(f"pytest framework source changed before accounting: {path}")
        expected_pins = {
            (item["module"], item["version"], item["path"], item["sha256"])
            for item in self.dependency_inventory
            if item["module"] in {"iniconfig", "packaging", "pluggy", "pygments", "pytest"}
        }
        inventory_pins = {
            (
                item.get("module"), item.get("version"), item.get("module_path"), item.get("module_sha256")
            )
            for item in inventory["distributions"]
            if isinstance(item, dict)
        }
        if expected_pins != inventory_pins or len(inventory_pins) != 5:
            raise GovernorError("pytest versions/source pins differ between inventories")

    def _append_role_not_started(self, phase: str, role: str, reason: str) -> None:
        if role in self.phase_dispositions:
            raise GovernorError("role already received a disposition")
        self.ledger.append(
            "ROLE_NOT_STARTED",
            phase=phase,
            role=role,
            status="not_started",
            reason=reason,
        )
        self.phase_dispositions[role] = "not_started"

    def _start_phase(self, phase: dict[str, Any]) -> None:
        now = time.monotonic_ns()
        phase_id = phase["phase_id"]
        remaining = self.hard_deadline_ns - now
        if now >= self.soft_deadline_ns or phase["subtree_reserved_seconds"] * 1_000_000_000 + 60_000_000_000 > remaining:
            raise GovernorError("full descendant reservation plus cleanup does not fit root deadline")
        self._verify_source_binding()
        roles = _all_roles(phase)
        validate_role_inventory(self.model, phase_id, roles)
        self.ledger.append(
            "PHASE_RESERVE",
            phase=phase_id,
            reserved_seconds=phase["subtree_reserved_seconds"],
            roles=roles,
        )
        self.phase_reservations.append(
            {
                "phase_id": phase_id,
                "reserved_seconds": phase["subtree_reserved_seconds"],
                "roles": roles,
            }
        )
        self.active_phase = phase_id
        self.phase_start_ns = now
        self.phase_deadline_ns = min(
            now + phase["wall_cap_seconds"] * 1_000_000_000,
            self.soft_deadline_ns,
        )
        self.phase_roles = {item["role"]: item for item in roles}
        declared_roles = phase.get("roles")
        controller_roles = (
            [item["role"] for item in declared_roles if item.get("kind") == "controller"]
            if isinstance(declared_roles, list)
            else []
        )
        if len(controller_roles) > 1:
            raise GovernorError("phase declares more than one root-broker controller")
        self.phase_controller_role = controller_roles[0] if controller_roles else None
        self.phase_dispositions = {}
        self.phase_started = {}
        (self.output_root / "phases" / phase_id).mkdir(parents=True, mode=0o700)

    def _finish_phase(
        self,
        phase: dict[str, Any],
        status: str,
        *,
        controller_result: dict[str, Any] | None = None,
    ) -> str:
        phase_id = phase["phase_id"]
        controller = self.phase_controller_role
        for item in _all_roles(phase):
            role = item["role"]
            if role not in self.phase_dispositions:
                job = self.jobs[(phase_id, role)]
                reason = "unused capacity" if job["entry_path"] is None else "controller did not request role"
                self._append_role_not_started(phase_id, role, reason)
        if controller is not None:
            children = [
                {
                    "role": item["role"],
                    "status": (
                        "success"
                        if self.phase_dispositions[item["role"]] == "success"
                        else "not_started"
                        if self.phase_dispositions[item["role"]] == "not_started"
                        else "failure"
                    ),
                    "reaped": True,
                }
                for item in _all_roles(phase)
                if item["role"] != controller
            ]
            controller_evidence = (
                controller_result.get("controller_terminal")
                if isinstance(controller_result, dict)
                else None
            )
            if not isinstance(controller_evidence, dict):
                status = "failed"
            else:
                try:
                    validate_controller_terminal(controller_evidence, children)
                except LedgerError:
                    status = "failed"
                if controller_evidence.get("status") != "success":
                    status = "failed"
        required_roles = {
            "A2": {
                "environment": {"environment_leaf"},
                "accounting": {
                    "accounting_test_controller",
                    "accounting_timer_fixture",
                    "accounting_pdeath_refusal",
                    "accounting_handshake_leaf",
                    "accounting_cleanup_leaf",
                },
                "retained": {"retained_leaf"},
                "metadata": {"metadata_controller", "metadata_worker"},
                "timing": {"timing_controller", "timing_worker"},
                "terminal": {"record_leaf"},
            },
            "Cylinder": {},
        }[self.model].get(phase_id)
        if required_roles is None:
            required_roles = (
                {"environment_leaf"}
                if phase_id == "preflight"
                else {"regression_worker", "accounting_controller_leaf", "contained_probe_helper"}
                if phase_id == "regression-1"
                else {"regression_worker"}
                if phase_id.startswith("regression-")
                else {"row_worker"}
                if phase_id.startswith("row-")
                else {"record_leaf"}
            )
        if any(self.phase_dispositions.get(role) != "success" for role in required_roles):
            status = "failed"
        elapsed = max(1, time.monotonic_ns() - self.phase_start_ns) / 1_000_000_000
        waited = [item["role"] for item in _all_roles(phase) if item["role"] != controller]
        if not self.phase_roles:
            raise GovernorError("phase lost its fixed role inventory")
        self.ledger.append(
            "PHASE_TERMINAL",
            phase=phase_id,
            status=status,
            waited_for_roles=waited,
            wait_seconds=elapsed,
        )
        summary = {
            "schema": "gragra-diagnostic-phase-result-v1",
            "model": self.model,
            "epoch_id": self.epoch_id,
            "phase_id": phase_id,
            "status": status,
            "roles": self.phase_dispositions,
            "elapsed_seconds": elapsed,
        }
        _write_new(self.output_root / "phases" / phase_id / "result.json", _canonical(summary))
        self.active_phase = None
        self.phase_roles = {}
        self.phase_controller_role = None
        self.phase_started = {}
        return status

    def _job_for(self, phase: str, role: str) -> dict[str, Any]:
        try:
            return self.jobs[(phase, role)]
        except KeyError as exc:
            raise GovernorError("root broker asked for an unreserved runtime job") from exc

    def _new_child_binding(self, phase: str, role: str, token: str, pid: int) -> dict[str, Any]:
        identity = process_identity(pid, require_argv=True)
        expected_argv = [
            self.python_path,
            "-I",
            "-S",
            self.shared_pins["bootstrap"]["path"],
            "--token",
            token,
        ]
        if identity["argv"] != expected_argv or identity["ppid"] != self.root["pid"]:
            raise GovernorError("Popen child argv or direct-parent identity differs")
        return {
            "pid": identity["pid"],
            "parent_pid": identity["ppid"],
            "boot_id": identity["boot_id"],
            "starttime_ticks": identity["starttime_ticks"],
            "parent_boot_id": self.root_boot_id,
            "parent_starttime_ticks": self.root["starttime_ticks"],
            "pdeath_signal": 9,
            "argv": identity["argv"],
            "launch_token": token,
            "phase": phase,
            "model": self.model,
            "ledger_id": self.ledger_id,
        }

    def _packet(self, phase: str, role: str, token: str, job: dict[str, Any], deadline_ns: int) -> dict[str, Any]:
        return {
            "model": self.model,
            "epoch_id": self.epoch_id,
            "ledger_id": self.ledger_id,
            "phase": phase,
            "role": role,
            "launch_token": token,
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "root_binding": {
                "pid": self.root["pid"],
                "boot_id": self.root_boot_id,
                "starttime_ticks": self.root["starttime_ticks"],
                "stat_identity": self.root_stat_identity,
                "argv": self.root.get("expected_argv", self.root.get("argv")),
            },
            "child_binding": None,
            "max_seconds": self.phase_roles[role]["max_seconds"],
            "effective_deadline_mono_ns": deadline_ns,
            "entry_path": str(Path(job["entry_path"]).resolve(strict=True)),
            "entry_sha256": job["entry_sha256"],
            "worktree_root": str(self.worktree_root),
            "site_packages": [str(Path(item).resolve(strict=True)) for item in self.site_packages],
            "result_path": job["result_path"],
            "source_catalog": self.catalog,
        }

    def _write_payload_after_admit(self, proc: subprocess.Popen[bytes], payload: dict[str, Any], deadline_ns: int) -> None:
        if proc.stdin is None:
            raise GovernorError("child stdin pipe is unavailable")
        raw = _canonical(payload)
        if len(raw) > 1_048_576:
            raise GovernorError("single worker payload exceeds fixed 1 MiB limit")
        fd = proc.stdin.fileno()
        os.set_blocking(fd, False)
        view = memoryview(raw)
        while view:
            self._check_root_deadline()
            if time.monotonic_ns() >= deadline_ns:
                raise GovernorError("role deadline expired while delivering admitted stdin")
            try:
                count = os.write(fd, view)
                if count <= 0:
                    raise GovernorError("worker stdin short write")
                view = view[count:]
            except BlockingIOError:
                select.select([], [fd], [], 0.25)
        proc.stdin.close()

    def _check_root_deadline(self) -> None:
        if time.monotonic_ns() >= self.soft_deadline_ns:
            self.cleanup_started = True
            raise GovernorError("root soft deadline reached; cleanup window is active")

    def _check_containment(self, hello: dict[str, Any], child_identity: dict[str, Any]) -> None:
        containment = hello.get("containment")
        if not isinstance(containment, dict):
            raise GovernorError("bootstrap containment proof is missing")
        expected_denied = {
            "clone": 56,
            "fork": 57,
            "vfork": 58,
            "execve": 59,
            "setuid": 105,
            "setgid": 106,
            "setreuid": 113,
            "setregid": 114,
            "setgroups": 116,
            "setresuid": 117,
            "setresgid": 119,
            "setfsuid": 122,
            "setfsgid": 123,
            "capset": 126,
            "prctl": 157,
            "unshare": 272,
            "setns": 308,
            "execveat": 322,
            "clone3": 435,
        }
        if (
            containment.get("architecture") != "x86_64"
            or containment.get("audit_architecture") != 0xC000003E
            or containment.get("denied_syscalls") != expected_denied
            or containment.get("no_new_privs") != 1
            or containment.get("seccomp_mode") != 2
            or child_identity.get("NoNewPrivs") != "1"
            or child_identity.get("Seccomp") != "2"
        ):
            raise GovernorError("child kernel no-fork/exec/credential containment is unproven")
        raw = base64.b64decode(containment.get("status_base64", ""), validate=True)
        if hashlib.sha256(raw).hexdigest() != containment.get("status_sha256"):
            raise GovernorError("child seccomp status evidence hash differs")
        if b"NoNewPrivs:\t1\n" not in raw or b"Seccomp:\t2\n" not in raw:
            raise GovernorError("child seccomp status bytes contradict readback")

    def _start_child(
        self, phase: str, role: str, *, requesting_controller: str | None = None
    ) -> dict[str, Any] | None:
        if self.host_admission.evidence.get("status") != "acquired":
            raise GovernorError("manifest-bound host worker admission is not held")
        if role in self.phase_dispositions or role in self.phase_started:
            raise GovernorError("root broker received a repeated role request")
        job = self._job_for(phase, role)
        if self.phase_controller_role is not None:
            if role == self.phase_controller_role and requesting_controller is not None:
                raise GovernorError("controller role cannot be requested as its own child")
            if role != self.phase_controller_role and requesting_controller != self.phase_controller_role:
                raise GovernorError("controller phase children may start only through the named broker")
            if (
                role != self.phase_controller_role
                and job["logical_controller_role"] != self.phase_controller_role
            ):
                raise GovernorError("requested role is not assigned to this phase controller")
        elif requesting_controller is not None:
            raise GovernorError("phase has no declared controller for this role request")
        if (
            self.model == "Cylinder"
            and phase == "regression-1"
            and role == "contained_probe_helper"
            and self.phase_dispositions.get("accounting_controller_leaf") != "success"
        ):
            self._append_role_not_started(
                phase,
                role,
                "contained probe requires a successful accounting_controller_leaf terminal",
            )
            return None
        if job["entry_path"] is None:
            self._append_role_not_started(phase, role, "unused capacity in frozen runtime_jobs")
            return None
        worker_payload = bind_root_manifest_sha256(job["stdin_payload"], self.manifest_sha256)
        if time.monotonic_ns() >= self.phase_deadline_ns:
            self._append_role_not_started(phase, role, "phase deadline expired before launch")
            return None
        self._verify_source_binding()
        role_max = self.phase_roles[role]["max_seconds"]
        now = time.monotonic_ns()
        role_max_deadline = min(
            now + role_max * 1_000_000_000,
            self.phase_deadline_ns - 10_000_000_000,
            self.hard_deadline_ns - 60_000_000_000,
        )
        if requesting_controller is not None:
            controller = self.phase_started.get(requesting_controller)
            if controller is None or controller["process"].poll() is not None:
                raise GovernorError("logical controller is absent or already terminal")
            role_max_deadline = min(role_max_deadline, controller["deadline_ns"] - 10_000_000_000)
        shutdown_allowance_ns = min(2_000_000_000, max(0, role_max_deadline - now))
        if shutdown_allowance_ns < _ROLE_KILL_REAP_RESERVE_NS:
            self._append_role_not_started(
                phase,
                role,
                "insufficient reserved lifetime remains for bounded SIGKILL/reap",
            )
            return None
        role_deadline = role_max_deadline - shutdown_allowance_ns
        if role_deadline <= now:
            self._append_role_not_started(phase, role, "no admitted role lifetime remains")
            return None
        token = secrets.token_hex(32)
        packet = self._packet(phase, role, token, job, role_deadline)
        prepare_record = {
            "schema": "gragra-diagnostic-launch-prepare-v1",
            "packet_without_child_binding": packet,
            "packet_sha256": sha256_bytes(_canonical(packet)),
            "reserved_role": self.phase_roles[role],
            "created_monotonic_ns": now,
        }
        _write_new(self.output_root / "launches" / f"{token}.prepare.json", _canonical(prepare_record))

        result_path = Path(job["result_path"])
        result_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if result_path.exists() or result_path.is_symlink():
            raise GovernorError("runtime result path is not fresh")
        stdout_path = self.output_root / "processes" / f"{token}.stdout"
        stderr_path = self.output_root / "processes" / f"{token}.stderr"
        stdout_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        stdout_file = stdout_path.open("xb")
        stderr_file = stderr_path.open("xb")
        parent_socket, child_socket = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        parent_socket.setblocking(True)
        env = dict(self.environment)
        env.update(
            {
                "GRAGRA_DIAGNOSTIC_CONTROL_FD": str(child_socket.fileno()),
                "GRAGRA_DIAGNOSTIC_MODEL": self.model,
                "GRAGRA_DIAGNOSTIC_PHASE": phase,
                "GRAGRA_DIAGNOSTIC_LEDGER_ID": self.ledger_id,
            }
        )
        argv = [
            self.python_path,
            "-I",
            "-S",
            self.shared_pins["bootstrap"]["path"],
            "--token",
            token,
        ]
        self._assert_root_single_threaded_identity()
        popen_start_ns = time.monotonic_ns()
        try:
            proc = subprocess.Popen(
                argv,
                cwd=str(self.worktree_root),
                env=env,
                stdin=subprocess.PIPE,
                stdout=stdout_file,
                stderr=stderr_file,
                close_fds=True,
                pass_fds=(child_socket.fileno(),),
                start_new_session=True,
                preexec_fn=make_pdeath_preexec(self.root["pid"]),
            )
        except BaseException:
            parent_socket.close()
            child_socket.close()
            stdout_file.close()
            stderr_file.close()
            raise
        finally:
            if child_socket.fileno() >= 0:
                child_socket.close()
        unbound = {
            "process": proc,
            "pid": proc.pid,
            "token": token,
            "phase": phase,
            "role": role,
            "parent_socket": parent_socket,
            "stdout_file": stdout_file,
            "stderr_file": stderr_file,
            "identity": None,
            "popen_start_mono_ns": popen_start_ns,
            "role_deadline_ns": role_deadline,
            "role_max_deadline_ns": role_max_deadline,
            "role_start_mono_ns": now,
            "shutdown_allowance_ns": shutdown_allowance_ns,
        }
        self.unbound_children.append(unbound)
        self._assert_root_single_threaded_identity()
        if proc.stdin is None:
            raise GovernorError("Popen did not provide child stdin")
        identity = process_identity(proc.pid, require_argv=True)
        unbound["identity"] = {
            "pid": identity["pid"],
            "starttime_ticks": identity["starttime_ticks"],
            "pgrp": identity["pgrp"],
            "boot_id": identity["boot_id"],
            "argv": identity["argv"],
        }
        self.declared.append(
            {
                "pid": identity["pid"],
                "starttime_ticks": identity["starttime_ticks"],
                "pgrp": identity["pgrp"],
                "boot_id": identity["boot_id"],
                "argv": identity["argv"],
                "phase": phase,
                "role": role,
                "launch_token": token,
            }
        )
        self._check_root_deadline()
        child_binding = self._new_child_binding(phase, role, token, proc.pid)
        handle = {
            "process": proc,
            "socket": parent_socket,
            "stdout_file": stdout_file,
            "stderr_file": stderr_file,
            "stdout_path": stdout_path,
            "stderr_path": stderr_path,
            "token": token,
            "phase": phase,
            "role": role,
            "job": job,
            "deadline_ns": role_deadline,
            "role_max_deadline_ns": role_max_deadline,
            "role_start_mono_ns": now,
            "shutdown_allowance_ns": shutdown_allowance_ns,
            "start_mono_ns": popen_start_ns,
            "child_binding": child_binding,
            "timer_before": None,
            "timer_observed": None,
            "timer_observed_ns": None,
            "rss_limit_fired": False,
            "root_timeout_fired": False,
            "cleanup_signal_sent": False,
            "control_buffer": bytearray(),
        }
        self.phase_started[role] = handle
        self.all_handles[token] = handle
        self.unbound_children.remove(unbound)
        if (self.model, phase, role) == ("A2", "accounting", "accounting_timer_fixture"):
            handle["timer_before"] = _proc_timer_raw(proc.pid)
        prepare = _read_bytes(self.output_root / "launches" / f"{token}.prepare.json")
        packet["child_binding"] = child_binding
        try:
            hello = _parse_control_line(parent_socket, min(role_deadline, time.monotonic_ns() + 10_000_000_000))
            if hello is None:
                raise GovernorError("child HELLO handshake exceeded its fixed 10 second limit")
            if hello.get("type") != "HELLO" or hello.get("launch_token") != token:
                raise GovernorError("child HELLO type/token differs")
            observed_binding = hello.get("binding")
            if not isinstance(observed_binding, dict):
                raise GovernorError("child HELLO binding is missing")
            validate_launch_binding(child_binding, observed_binding)
            if hello.get("parent_identity") != self.root_stat_identity:
                raise GovernorError("child observed a different root parent identity")
            child_identity = process_identity(proc.pid, require_argv=True)
            self._check_containment(hello, child_identity)
            hello_record = {
                "schema": "gragra-diagnostic-child-hello-v1",
                "launch_token": token,
                "binding": observed_binding,
                "parent_identity": hello["parent_identity"],
                "containment": hello["containment"],
                "packet_prepare_sha256": hashlib.sha256(prepare).hexdigest(),
            }
            _write_new(self.output_root / "processes" / f"{token}-hello.json", _canonical(hello_record))
            start_event = {
                key: value for key, value in observed_binding.items()
                if key not in _LEDGER_OWNED_FIELDS
            }
            start_event.update(
                {
                    "role_start_monotonic_ns": handle["role_start_mono_ns"],
                    "role_deadline_monotonic_ns": handle["deadline_ns"],
                    "role_max_deadline_monotonic_ns": handle["role_max_deadline_ns"],
                    "shutdown_allowance_ns": handle["shutdown_allowance_ns"],
                }
            )
            self.ledger.append("CHILD_START", **start_event)
            admitted_packet = {
                **packet,
                "child_binding": observed_binding,
            }
            final_launch = {
                "schema": "gragra-diagnostic-launch-v1",
                "packet": admitted_packet,
                "packet_sha256": sha256_bytes(_canonical(admitted_packet)),
                "prepare_sha256": hashlib.sha256(prepare).hexdigest(),
            }
            _write_new(self.output_root / "launches" / f"{token}.json", _canonical(final_launch))
            _send_control(
                parent_socket,
                {"type": "ADMIT", "launch_token": token, "launch_packet": admitted_packet},
            )
            self._write_payload_after_admit(proc, worker_payload, role_deadline)
            parent_socket.setblocking(False)
            handle["hello"] = hello_record
            handle["packet"] = admitted_packet
            return handle
        except BaseException as exc:
            try:
                _send_control(
                    parent_socket,
                    {"type": "REFUSED", "launch_token": token, "reason": str(exc)},
                )
            except BaseException:
                pass
            self._terminate_handle(handle, signal.SIGKILL)
            self._reap_handle(handle, "failed")
            parent_socket.close()
            raise

    def _assert_root_single_threaded_identity(self) -> None:
        current = root_identity()
        fields = (
            "pid",
            "ppid",
            "pgrp",
            "session",
            "starttime_ticks",
            "boot_id",
            "argv",
            "expected_argv",
        )
        if any(current.get(field) != self.root.get(field) for field in fields):
            raise GovernorError("root identity or single-threaded preexec condition changed")

    def _sample_rss(self) -> None:
        now = time.monotonic_ns()
        if now < self.rss_next_sample_ns:
            return
        identities = enumerate_processes()
        root_and_direct = {
            identity["pid"]: identity
            for identity in identities
            if identity["pid"] == self.root["pid"] or identity["ppid"] == self.root["pid"]
        }
        expected = {self.root["pid"]}
        expected.update(item["pid"] for item in self.declared)
        unexpected = set(root_and_direct) - expected
        if unexpected:
            raise GovernorError(f"unregistered direct child/process identity observed: {sorted(unexpected)}")
        for handle in self.phase_started.values():
            identity = root_and_direct.get(handle["process"].pid)
            if identity is not None:
                if identity["NoNewPrivs"] != "1" or identity["Seccomp"] != "2" or identity["threads"] != 1:
                    raise GovernorError("managed process containment/thread identity changed")
        if self.cgroup is not None:
            self.cgroup_sample = self.cgroup.sample()
            rss = self.cgroup_sample["current_bytes"]
        else:
            rss = sum(identity["rss_bytes"] for identity in root_and_direct.values())
        previous_peak = self.rss_peak_bytes
        self.rss_peak_bytes = max(previous_peak, rss)
        if rss > previous_peak:
            self.rss_peak_mono_ns = now
        self.rss_sample_count += 1
        self.rss_next_sample_ns = now + 1_000_000_000
        if self.cgroup is None and rss > 4 * 1024 * 1024 * 1024:
            for handle in list(self.phase_started.values()):
                handle["rss_limit_fired"] = True
                self._terminate_handle(handle, signal.SIGKILL)
            raise GovernorError("soft process-tree RSS observation exceeded 4 GiB")

    def _observe_timer_fixture(self, handle: dict[str, Any]) -> None:
        if handle["timer_before"] is None or handle["timer_observed"] is not None:
            return
        raw = _proc_timer_raw(handle["process"].pid)
        before_records = parse_proc_timers(handle["timer_before"])
        records = parse_proc_timers(raw)
        added = set(records) - set(before_records)
        if not added:
            return
        record = verify_new_proc_timer(
            handle["timer_before"], raw, root_pid=handle["process"].pid
        )
        handle["timer_observed"] = record
        handle["timer_after"] = raw
        handle["timer_observed_ns"] = time.monotonic_ns()

    def _record_role_maximum_event(
        self,
        handle: dict[str, Any],
        *,
        state: str,
        observed_terminal_ns: int | None,
        detection_mono_ns: int,
        cleanup_unconfirmed: bool,
    ) -> dict[str, Any]:
        role_deadline = handle["deadline_ns"]
        role_max = handle["role_max_deadline_ns"]
        event = {
            "phase": handle["phase"],
            "role": handle["role"],
            "pid": handle["process"].pid,
            "state": state,
            "role_deadline_monotonic_ns": role_deadline,
            "role_max_deadline_monotonic_ns": role_max,
            "terminal_observed_monotonic_ns": observed_terminal_ns,
            "detection_monotonic_ns": detection_mono_ns,
            "dispatch_lateness_ns": (
                None
                if observed_terminal_ns is None
                else max(0, observed_terminal_ns - role_deadline)
            ),
            "maximum_overrun_ns": (
                None
                if observed_terminal_ns is None
                else max(0, observed_terminal_ns - role_max)
            ),
            "deadline_overrun_at_detection_ns": max(0, detection_mono_ns - role_max),
            "cleanup_unconfirmed": cleanup_unconfirmed,
        }
        self.role_maximum_events.append(event)
        if cleanup_unconfirmed:
            handle["role_maximum_wait_unconfirmed"] = True
        return event

    def _mark_role_maximum_wait_unconfirmed(self, handle: dict[str, Any]) -> None:
        if handle.get("role_maximum_wait_unconfirmed"):
            return
        now = time.monotonic_ns()
        handle["role_maximum_expired"] = True
        self._record_role_maximum_event(
            handle,
            state="cleanup_unconfirmed_at_role_maximum",
            observed_terminal_ns=None,
            detection_mono_ns=now,
            cleanup_unconfirmed=True,
        )

    def _store_wait_evidence(self, handle: dict[str, Any], return_code: int, *, basis: str) -> int:
        if type(return_code) is not int:
            raise GovernorError("native Popen wait returned a non-integer status")
        expected_wait_status = -return_code if return_code < 0 else return_code << 8
        previous_code = handle.get("returncode")
        previous_wait = handle.get("wait_status")
        if previous_code is not None and previous_code != return_code:
            raise GovernorError("native Popen return code changed after wait")
        if previous_wait is not None and previous_wait != expected_wait_status:
            raise GovernorError("native wait status changed after wait")
        observed_ns = handle.get("terminal_mono_ns")
        if observed_ns is None:
            observed_ns = time.monotonic_ns()
            handle["terminal_mono_ns"] = observed_ns
        elif type(observed_ns) is not int or observed_ns <= 0:
            raise GovernorError("native terminal observation time is malformed")
        if not isinstance(handle.get("terminal_observation_basis"), str):
            handle["terminal_observation_basis"] = basis
        role_deadline = handle.get("deadline_ns")
        role_max = handle.get("role_max_deadline_ns", role_deadline)
        if type(role_deadline) is not int or type(role_max) is not int:
            raise GovernorError("role terminal deadlines are malformed")
        late_by_ns = max(0, observed_ns - role_deadline)
        maximum_overrun_ns = max(0, observed_ns - role_max)
        handle["late_terminal"] = bool(handle.get("late_terminal") or late_by_ns > 0)
        handle["late_by_ns"] = late_by_ns
        handle["dispatch_late_by_ns"] = late_by_ns
        handle["maximum_overrun_ns"] = maximum_overrun_ns
        handle["maximum_overrun"] = maximum_overrun_ns > 0
        handle["returncode"] = return_code
        handle["wait_status"] = expected_wait_status
        if maximum_overrun_ns > 0 and not handle.get("role_maximum_terminal_recorded"):
            handle["role_maximum_expired"] = True
            handle["role_maximum_terminal_recorded"] = True
            self._record_role_maximum_event(
                handle,
                state="terminal_observed_after_role_maximum",
                observed_terminal_ns=observed_ns,
                detection_mono_ns=time.monotonic_ns(),
                cleanup_unconfirmed=False,
            )
        return return_code

    def _wait_process(self, handle: dict[str, Any], *, force_terminate: bool = False) -> int:
        proc = handle["process"]
        role_max = handle.get("role_max_deadline_ns", handle["deadline_ns"])
        shutdown_allowance = handle.get("shutdown_allowance_ns", 0)
        if (
            type(role_max) is not int
            or type(shutdown_allowance) is not int
            or shutdown_allowance < _ROLE_KILL_REAP_RESERVE_NS
        ):
            raise GovernorError("role shutdown envelope is malformed")

        observed = proc.poll()
        if observed is not None:
            try:
                return_code = proc.wait(timeout=0)
            except subprocess.TimeoutExpired as exc:
                raise GovernorError("Popen poll exit could not be normalized by bounded wait") from exc
            if return_code != observed:
                raise GovernorError("Popen poll and bounded wait disagree")
            return self._store_wait_evidence(handle, return_code, basis="popen_poll_after_exit")

        terminate = force_terminate
        while not terminate:
            observed = proc.poll()
            if observed is not None:
                try:
                    return_code = proc.wait(timeout=0)
                except subprocess.TimeoutExpired as exc:
                    raise GovernorError("Popen poll exit could not be normalized by bounded wait") from exc
                if return_code != observed:
                    raise GovernorError("Popen poll and bounded wait disagree")
                return self._store_wait_evidence(handle, return_code, basis="popen_poll_after_exit")
            now = time.monotonic_ns()
            if now >= handle["deadline_ns"]:
                handle["role_timeout_fired"] = True
                terminate = True
                break
            if now >= role_max:
                handle["role_timeout_fired"] = True
                handle["role_maximum_expired"] = True
                terminate = True
                break
            if now >= self.soft_deadline_ns:
                self.cleanup_started = True
                handle["root_timeout_fired"] = True
                self._terminate_handle(handle, signal.SIGKILL)
                handle["kill_sent_monotonic_ns"] = time.monotonic_ns()
                break
            self._observe_timer_fixture(handle)
            self._sample_rss()
            until_deadline = max(0, handle["deadline_ns"] - time.monotonic_ns())
            time.sleep(min(_ROLE_POLL_INTERVAL_NS, until_deadline) / 1_000_000_000)

        if proc.poll() is None and terminate and not handle.get("kill_sent_monotonic_ns"):
            if not handle.get("term_sent_monotonic_ns"):
                self._terminate_handle(handle, signal.SIGTERM)
            grace_end = min(
                time.monotonic_ns() + shutdown_allowance - _ROLE_KILL_REAP_RESERVE_NS,
                role_max - _ROLE_KILL_REAP_RESERVE_NS,
            )
            while proc.poll() is None and time.monotonic_ns() < grace_end:
                if time.monotonic_ns() >= self.soft_deadline_ns:
                    self.cleanup_started = True
                    handle["root_timeout_fired"] = True
                    break
                self._sample_rss()
                time.sleep(
                    min(
                        _ROLE_POLL_INTERVAL_NS,
                        max(0, grace_end - time.monotonic_ns()),
                    )
                    / 1_000_000_000
                )
            if proc.poll() is None:
                self._terminate_handle(handle, signal.SIGKILL)
                handle["kill_sent_monotonic_ns"] = time.monotonic_ns()

        observed = proc.poll()
        if observed is not None:
            try:
                return_code = proc.wait(timeout=0)
            except subprocess.TimeoutExpired as exc:
                raise GovernorError("Popen poll exit could not be normalized by bounded wait") from exc
            if return_code != observed:
                raise GovernorError("Popen poll and bounded wait disagree")
            basis = "wait_after_termination" if terminate else "popen_poll_after_exit"
            return self._store_wait_evidence(handle, return_code, basis=basis)

        remaining_ns = max(0, role_max - time.monotonic_ns())
        if remaining_ns > 0:
            try:
                return_code = proc.wait(timeout=remaining_ns / 1_000_000_000)
            except subprocess.TimeoutExpired:
                return_code = None
            except (OSError, ChildProcessError) as exc:
                self._mark_role_maximum_wait_unconfirmed(handle)
                raise GovernorError("native bounded child wait failed; cleanup remains unconfirmed") from exc
        else:
            return_code = None
        if return_code is None:
            observed = proc.poll()
            if observed is not None:
                try:
                    return_code = proc.wait(timeout=0)
                except subprocess.TimeoutExpired:
                    return_code = None
                if return_code != observed:
                    return_code = None
        if return_code is None:
            self._mark_role_maximum_wait_unconfirmed(handle)
            raise GovernorError("child remained unreaped at its reserved role maximum")
        basis = "wait_after_termination" if terminate else "popen_wait"
        return self._store_wait_evidence(handle, return_code, basis=basis)

    def _capture_wait_evidence(self, handle: dict[str, Any]) -> int:
        """Normalize native wait fields even when poll() observed exit first."""
        proc = handle["process"]
        polled = proc.poll()
        if polled is None:
            return self._wait_process(
                handle, force_terminate=bool(handle.get("cleanup_signal_sent", False))
            )
        try:
            return_code = proc.wait(timeout=0)
        except (OSError, ChildProcessError) as exc:
            raise GovernorError("already-exited child could not be positively reaped") from exc
        if return_code != polled:
            raise GovernorError("Popen poll and wait disagree for an exited child")
        basis = handle.get("terminal_observation_basis", "popen_poll_after_exit")
        return self._store_wait_evidence(handle, return_code, basis=basis)

    def _reap_handle_popen_evidence(self, handle: dict[str, Any]) -> dict[str, Any]:
        """Bind Popen's completed wait to the original PID/starttime lifetime."""
        proc = handle["process"]
        return_code = handle.get("returncode")
        polled = proc.poll()
        if return_code is None or polled is None or polled != return_code or proc.returncode != return_code:
            raise GovernorError("Popen wait/poll terminal values are missing or contradictory")
        expected_wait_status = -return_code if return_code < 0 else return_code << 8
        if handle.get("wait_status") != expected_wait_status:
            raise GovernorError("stored native wait status contradicts Popen return code")
        binding = handle["child_binding"]
        try:
            current = process_identity(binding["pid"], require_argv=False)
        except ProcfsGone:
            current = None
        except ProcfsError as exc:
            raise GovernorError("post-wait proc identity is unavailable rather than vanished") from exc
        if current is not None and (
            current.get("boot_id") == binding["boot_id"]
            and current.get("starttime_ticks") == binding["starttime_ticks"]
        ):
            raise GovernorError("Popen reports wait completion while original PID lifetime remains visible")
        evidence = {
            "schema": "gragra-diagnostic-popen-reap-evidence-v1",
            "pid": binding["pid"],
            "starttime_ticks": binding["starttime_ticks"],
            "boot_id": binding["boot_id"],
            "popen_poll_returncode": polled,
            "popen_returncode": return_code,
            "native_wait_status": handle["wait_status"],
            "wait_completed": True,
            "original_identity_absent": True,
            "replacement_pid_identity": current,
            "terminal_monotonic_ns": handle["terminal_mono_ns"],
        }
        handle["popen_reap_evidence"] = evidence
        return evidence

    def _drive_controller(self, handle: dict[str, Any]) -> None:
        """Broker only pre-reserved direct-root children named by this controller."""
        proc = handle["process"]
        seen_requests: set[str] = set()
        expected_roles = [
            role for role in self.phase_roles if role != handle["role"]
        ]
        seen_roles: list[str] = []
        while True:
            self._check_root_deadline()
            now = time.monotonic_ns()
            if now >= handle["deadline_ns"]:
                self._wait_process(handle)
                return
            self._sample_rss()
            try:
                message = _receive_controller_record(
                    handle, min(handle["deadline_ns"], now + 250_000_000)
                )
            except ControllerBrokerEOF as exc:
                try:
                    validate_controller_broker_eof(expected_roles, seen_roles)
                except LedgerError as validation_error:
                    raise GovernorError(str(validation_error)) from exc
                return
            if message is None:
                if proc.poll() is not None:
                    raise GovernorError(
                        "controller exited before clean broker EOF was observed"
                    )
                continue
            if proc.poll() is not None:
                raise GovernorError(
                    "controller exited with an unprocessed broker request"
                )
            required = {"type", "launch_token", "phase_id", "role", "request_id"}
            request_id = message.get("request_id")
            role = message.get("role")
            if (
                set(message) != required
                or message.get("type") != "REQUEST_ROLE"
                or message.get("launch_token") != handle["token"]
                or message.get("phase_id") != handle["phase"]
                or not isinstance(request_id, str)
                or not request_id
                or len(request_id) > 128
                or request_id in seen_requests
                or not isinstance(role, str)
                or role not in expected_roles
                or role in seen_roles
            ):
                raise GovernorError("controller role request fields/identity are invalid")
            seen_requests.add(request_id)
            seen_roles.append(role)
            child_result: dict[str, Any]
            child = self._start_child(
                handle["phase"], role, requesting_controller=handle["role"]
            )
            if child is None:
                child_result = {
                    "status": "not_started",
                    "terminal_pin": None,
                    "result_pin": None,
                    "result_status": None,
                }
            else:
                self._wait_process(child)
                result = self._reap_handle(child, "terminal")
                child_result = {
                    "status": result["status"],
                    "terminal_pin": result["terminal_pin"],
                    "result_pin": result["result_pin"],
                    "result_status": (
                        result["result"].get("status")
                        if isinstance(result.get("result"), dict)
                        else None
                    ),
                }
            self._check_root_deadline()
            _send_control(
                handle["socket"],
                {
                    "type": "ROLE_RESULT",
                    "launch_token": handle["token"],
                    "phase_id": handle["phase"],
                    "role": role,
                    "request_id": request_id,
                    **child_result,
                },
            )

    def _terminate_handle(self, handle: dict[str, Any], sig: int) -> None:
        proc = handle["process"]
        if proc.poll() is not None:
            return
        try:
            current = process_identity(proc.pid, require_argv=True)
        except ProcfsError as exc:
            raise GovernorError("managed child identity disappeared before signal decision") from exc
        declared = next((item for item in self.declared if item["pid"] == proc.pid), None)
        if (
            declared is None
            or current["pid"] != declared["pid"]
            or current["starttime_ticks"] != declared["starttime_ticks"]
            or current["boot_id"] != declared["boot_id"]
            or current["ppid"] != self.root["pid"]
            or current["pgrp"] != declared["pgrp"]
            or current["argv"] != declared["argv"]
        ):
            raise GovernorError("managed child PID/starttime/parent/group/argv identity changed before signal")
        proc.send_signal(sig)
        if sig in {signal.SIGTERM, signal.SIGKILL}:
            handle["cleanup_signal_sent"] = True
        if sig == signal.SIGTERM:
            handle.setdefault("term_sent_monotonic_ns", time.monotonic_ns())
        elif sig == signal.SIGKILL:
            handle.setdefault("kill_sent_monotonic_ns", time.monotonic_ns())

    def _reap_handle(self, handle: dict[str, Any], status: str) -> dict[str, Any]:
        proc = handle["process"]
        return_code = self._capture_wait_evidence(handle)
        signal_number = -return_code if return_code < 0 else None
        exit_code = return_code if return_code >= 0 else None
        expected_terminal = handle["job"].get("expected_terminal")
        expected_timer = (
            self.model == "A2"
            and handle["phase"] == "accounting"
            and handle["role"] == "accounting_timer_fixture"
        )
        result = None
        result_raw = b""
        result_error = None
        timer_profile = {
            "profile": "a2-accounting-child-timer-sigkill-v1",
            "model": "A2",
            "phase": "accounting",
            "role": "accounting_timer_fixture",
            "signal": 9,
            "source": "child_owned_posix_timer",
        }
        if expected_timer and signal_number == 9:
            try:
                if expected_terminal != timer_profile:
                    raise GovernorError("timer fixture expected-terminal profile differs")
                if (
                    handle["timer_observed"] is None
                    or handle.get("timer_after") is None
                    or handle["root_timeout_fired"]
                    or handle.get("role_timeout_fired", False)
                    or handle["rss_limit_fired"]
                    or handle["cleanup_signal_sent"]
                    or handle["terminal_mono_ns"] >= handle["deadline_ns"]
                    or handle.get("late_terminal", False)
                    or handle["terminal_mono_ns"] >= self.soft_deadline_ns
                    or return_code != -9
                ):
                    raise GovernorError("child SIGKILL lacks the preobserved timer/deadline proof")
                evidence = {
                    "timer_raw_before_base64": base64.b64encode(handle["timer_before"]).decode("ascii"),
                    "timer_raw_after_base64": base64.b64encode(handle["timer_after"]).decode("ascii"),
                    "child_timer_record": handle["timer_observed"],
                    "timer_observed_monotonic_ns": handle["timer_observed_ns"],
                    "terminal_monotonic_ns": handle["terminal_mono_ns"],
                    "role_deadline_monotonic_ns": handle["deadline_ns"],
                    "role_max_deadline_monotonic_ns": handle["role_max_deadline_ns"],
                    "shutdown_allowance_ns": handle["shutdown_allowance_ns"],
                    "root_soft_deadline_monotonic_ns": self.soft_deadline_ns,
                    "root_deadline_monotonic_ns": self.hard_deadline_ns,
                    "child_timer_owned": True,
                    "native_wait_signal": 9,
                    "root_timeout_fired": False,
                    "rss_limit_fired": False,
                    "cleanup_signal_sent": False,
                }
                result = {
                    "schema": "gragra.a2.diagnostic.worker-result.v1",
                    "status": "PASS",
                    "diagnostic_only": True,
                    "science_credit": 0,
                    "expected_negative_terminal": True,
                }
                result_raw = _canonical(result)
                _write_new(Path(handle["job"]["result_path"]), result_raw)
                handle["expected_terminal_evidence"] = evidence
            except BaseException as exc:
                result_error = f"expected timer outcome proof failed: {exc}"
        else:
            try:
                stdout_raw = _read_bytes(handle["stdout_path"])
                if not stdout_raw.endswith(b"\n") or stdout_raw.count(b"\n") != 1:
                    raise GovernorError("family stdout is not exactly one JSON result line")
                result = strict_json(stdout_raw[:-1])
                if not isinstance(result, dict):
                    raise GovernorError("family result is not a JSON object")
                required_schema = (
                    "gragra.a2.diagnostic.worker-result.v1"
                    if self.model == "A2"
                    else "gragra.cylinder.diagnostic.worker-result.v1"
                )
                if (
                    result.get("schema") != required_schema
                    or result.get("status") not in {"PASS", "FAIL", "REFUSED"}
                    or result.get("diagnostic_only") is not True
                    or type(result.get("science_credit")) is not int
                    or result["science_credit"] != 0
                ):
                    raise GovernorError("family result schema/status/credit differs")
                result_path = Path(handle["job"]["result_path"])
                result_raw = _read_bytes(result_path)
                if result_raw != stdout_raw:
                    raise GovernorError("family result file differs byte-for-byte from stdout result")
            except BaseException as exc:
                result = None
                result_raw = b""
                result_error = str(exc)

        self._sample_rss()
        self._reap_handle_popen_evidence(handle)
        role_status = "success" if result is not None and result.get("status") == "PASS" else "failed"
        if exit_code == 0 and result is not None and result.get("status") == "REFUSED":
            role_status = "refused"
        elif exit_code == 0 and result is not None and result.get("status") == "FAIL":
            role_status = "failed"
        elif signal_number is not None and not expected_timer:
            role_status = "failed"
        if expected_timer and signal_number == 9 and handle.get("expected_terminal_evidence"):
            role_status = "success"
        if role_status == "success" and not expected_timer and exit_code != 0:
            role_status = "failed"
        if (
            status != "terminal"
            or handle.get("role_timeout_fired")
            or handle.get("root_timeout_fired")
            or handle.get("late_terminal")
        ):
            role_status = "failed"
        reaped_identity = {
            "pid": handle["child_binding"]["pid"],
            "starttime_ticks": handle["child_binding"]["starttime_ticks"],
            "wait_status": handle["wait_status"],
        }
        self.reaped.append(reaped_identity)
        terminal_result_status = (
            "PASS"
            if role_status == "success"
            else result.get("status")
            if isinstance(result, dict) and result.get("status") in {"FAIL", "REFUSED"}
            else "REFUSED"
        )
        terminal = {
            **handle["child_binding"],
            "role": handle["role"],
            "status": role_status,
            "result_status": terminal_result_status,
            "exit_code": exit_code,
            "signal": signal_number,
            "wait_status": handle["wait_status"],
            "popen_returncode": return_code,
            "popen_reap_evidence": handle["popen_reap_evidence"],
            "cleanup_confirmed": True,
            "reaped": True,
            "terminal_monotonic_ns": handle["terminal_mono_ns"],
            "terminal_observation_basis": handle["terminal_observation_basis"],
            "role_start_monotonic_ns": handle["role_start_mono_ns"],
            "role_deadline_monotonic_ns": handle["deadline_ns"],
            "role_max_deadline_monotonic_ns": handle["role_max_deadline_ns"],
            "shutdown_allowance_ns": handle["shutdown_allowance_ns"],
            "role_timeout_fired": bool(handle.get("role_timeout_fired", False)),
            "late_terminal": bool(handle.get("late_terminal", False)),
            "late_by_ns": handle.get("late_by_ns", 0),
            "dispatch_late_by_ns": handle.get("dispatch_late_by_ns", 0),
            "maximum_overrun_ns": handle.get("maximum_overrun_ns", 0),
            "maximum_overrun": bool(handle.get("maximum_overrun", False)),
            "cleanup_unconfirmed_at_role_maximum": bool(
                handle.get("role_maximum_wait_unconfirmed", False)
            ),
            "child_elapsed_seconds": max(0, handle["terminal_mono_ns"] - handle["start_mono_ns"]) / 1_000_000_000,
            "stdout_sha256": sha256_bytes(_read_bytes(handle["stdout_path"])),
            "stderr_sha256": sha256_bytes(_read_bytes(handle["stderr_path"])),
        }
        if result_error is not None:
            terminal["result_validation_error"] = result_error
        if expected_timer and handle.get("expected_terminal_evidence"):
            terminal["expected_terminal"] = expected_terminal
            terminal["expected_terminal_evidence"] = handle["expected_terminal_evidence"]
        terminal_path = self.output_root / "processes" / f"{handle['token']}-terminal.json"
        terminal_record = {
            "schema": "gragra-diagnostic-child-terminal-v1",
            "terminal": terminal,
            "result_sha256": sha256_bytes(result_raw) if result_raw else None,
        }
        _write_new(terminal_path, _canonical(terminal_record))
        terminal_event = {
            key: value for key, value in terminal.items()
            if key not in _LEDGER_OWNED_FIELDS
        }
        self.ledger.append("CHILD_TERMINAL", **terminal_event)
        handle["terminal"] = terminal
        handle["terminal_path"] = terminal_path
        handle["result"] = result
        handle["result_path"] = Path(handle["job"]["result_path"])
        self.phase_dispositions[handle["role"]] = role_status
        self.phase_started.pop(handle["role"], None)
        try:
            handle["socket"].close()
        except OSError:
            pass
        handle["stdout_file"].close()
        handle["stderr_file"].close()
        return {
            "status": role_status,
            "terminal_pin": {
                "path": str(terminal_path),
                "sha256": sha256_file(terminal_path),
            },
            "result_pin": (
                {"path": str(handle["result_path"]), "sha256": sha256_file(handle["result_path"])}
                if handle["result_path"].is_file()
                else None
            ),
            "result": result,
        }

    def _reap_unexpected_children(self) -> list[dict[str, Any]]:
        known = {item["pid"] for item in self.declared}
        return reap_untracked_waitables(known)

    def _close_unbound_children(self) -> None:
        for item in list(self.unbound_children):
            proc = item["process"]
            signals_sent: list[int] = []
            terminal_mono_ns = None
            late_by_ns = None
            late_terminal = False
            maximum_overrun_ns = None
            wait_unconfirmed = False
            for stream in (proc.stdin, item["parent_socket"]):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
            for stream in (item["stdout_file"], item["stderr_file"]):
                try:
                    stream.close()
                except OSError:
                    pass
            try:
                if proc.poll() is None:
                    try:
                        proc.send_signal(signal.SIGTERM)
                        signals_sent.append(signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    role_max = item["role_max_deadline_ns"]
                    grace_end = min(
                        time.monotonic_ns()
                        + item["shutdown_allowance_ns"]
                        - _ROLE_KILL_REAP_RESERVE_NS,
                        role_max - _ROLE_KILL_REAP_RESERVE_NS,
                    )
                    while proc.poll() is None and time.monotonic_ns() < grace_end:
                        time.sleep(
                            min(
                                _ROLE_POLL_INTERVAL_NS,
                                max(0, grace_end - time.monotonic_ns()),
                            )
                            / 1_000_000_000
                        )
                    if proc.poll() is None:
                        try:
                            proc.send_signal(signal.SIGKILL)
                            signals_sent.append(signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                return_code = proc.returncode
                if return_code is None:
                    remaining_ns = max(
                        0, item["role_max_deadline_ns"] - time.monotonic_ns()
                    )
                    if remaining_ns == 0:
                        raise subprocess.TimeoutExpired(proc.args, 0)
                    return_code = proc.wait(timeout=remaining_ns / 1_000_000_000)
                else:
                    return_code = proc.wait(timeout=0)
                terminal_mono_ns = time.monotonic_ns()
                late_by_ns = max(0, terminal_mono_ns - item["role_deadline_ns"])
                late_terminal = late_by_ns > 0
                maximum_overrun_ns = max(
                    0, terminal_mono_ns - item["role_max_deadline_ns"]
                )
                wait_status = (
                    -return_code if return_code < 0 else return_code << 8
                )
            except subprocess.TimeoutExpired:
                observed = proc.poll()
                if observed is not None:
                    try:
                        return_code = proc.wait(timeout=0)
                    except subprocess.TimeoutExpired:
                        return_code = None
                    if return_code == observed:
                        terminal_mono_ns = time.monotonic_ns()
                        late_by_ns = max(
                            0, terminal_mono_ns - item["role_deadline_ns"]
                        )
                        late_terminal = late_by_ns > 0
                        maximum_overrun_ns = max(
                            0, terminal_mono_ns - item["role_max_deadline_ns"]
                        )
                        wait_status = -return_code if return_code < 0 else return_code << 8
                    else:
                        return_code = None
                        wait_status = None
                        terminal_mono_ns = None
                        late_by_ns = None
                        late_terminal = False
                        maximum_overrun_ns = None
                        wait_unconfirmed = True
                else:
                    return_code = None
                    wait_status = None
                    terminal_mono_ns = None
                    late_by_ns = None
                    late_terminal = False
                    maximum_overrun_ns = None
                    wait_unconfirmed = True
            except (OSError, ChildProcessError) as exc:
                return_code = None
                wait_status = None
                terminal_mono_ns = None
                late_by_ns = None
                late_terminal = False
                maximum_overrun_ns = None
                wait_unconfirmed = True
                item["wait_error"] = f"{type(exc).__name__}: {exc}"
            if wait_unconfirmed:
                self.role_maximum_events.append(
                    {
                        "phase": item["phase"],
                        "role": item["role"],
                        "pid": item["pid"],
                        "state": "unbound_cleanup_unconfirmed_at_role_maximum",
                        "role_start_monotonic_ns": item["role_start_mono_ns"],
                        "role_deadline_monotonic_ns": item["role_deadline_ns"],
                        "role_max_deadline_monotonic_ns": item["role_max_deadline_ns"],
                        "terminal_observed_monotonic_ns": None,
                        "detection_monotonic_ns": time.monotonic_ns(),
                        "dispatch_lateness_ns": None,
                        "maximum_overrun_ns": None,
                        "cleanup_unconfirmed": True,
                    }
                )
            elif maximum_overrun_ns:
                self.role_maximum_events.append(
                    {
                        "phase": item["phase"],
                        "role": item["role"],
                        "pid": item["pid"],
                        "state": "unbound_terminal_observed_after_role_maximum",
                        "role_start_monotonic_ns": item["role_start_mono_ns"],
                        "role_deadline_monotonic_ns": item["role_deadline_ns"],
                        "role_max_deadline_monotonic_ns": item["role_max_deadline_ns"],
                        "terminal_observed_monotonic_ns": terminal_mono_ns,
                        "detection_monotonic_ns": time.monotonic_ns(),
                        "dispatch_lateness_ns": late_by_ns,
                        "maximum_overrun_ns": maximum_overrun_ns,
                        "cleanup_unconfirmed": False,
                    }
                )
            self.unbound_child_evidence.append(
                {
                    "pid_from_popen": item["pid"],
                    "token": item["token"],
                    "phase": item["phase"],
                    "role": item["role"],
                    "observed_identity": item["identity"],
                    "popen_start_mono_ns": item["popen_start_mono_ns"],
                    "cleanup_signals_sent": signals_sent,
                    "returncode": return_code,
                    "wait_status": wait_status,
                    "terminal_monotonic_ns": terminal_mono_ns,
                    "late_terminal": late_terminal,
                    "late_by_ns": late_by_ns,
                    "dispatch_late_by_ns": late_by_ns,
                    "maximum_overrun_ns": maximum_overrun_ns,
                    "maximum_overrun": bool(maximum_overrun_ns),
                    "cleanup_unconfirmed_at_role_maximum": wait_unconfirmed,
                    "role_start_monotonic_ns": item["role_start_mono_ns"],
                    "role_deadline_monotonic_ns": item["role_deadline_ns"],
                    "role_max_deadline_monotonic_ns": item["role_max_deadline_ns"],
                    "deadline_outcome": (
                        "wait_unconfirmed"
                        if terminal_mono_ns is None
                        else "maximum_overrun"
                        if maximum_overrun_ns
                        else "late"
                        if late_terminal
                        else "observed_within_role_deadline"
                    ),
                    "identity_and_terminal_binding_unconfirmed": True,
                }
            )
            self.unbound_children.remove(item)

    def _execute_phase(self, phase: dict[str, Any]) -> str:
        controller_result = None
        status = "success"
        try:
            self._start_phase(phase)
            if self.model == "A2" and phase["phase_id"] == "accounting":
                self._verify_test_framework_sources()
            if self.phase_controller_role is not None:
                controller_role = self.phase_controller_role
                handle = self._start_child(phase["phase_id"], controller_role)
                if handle is not None:
                    self._drive_controller(handle)
                    self._wait_process(handle)
                    outcome = self._reap_handle(handle, "terminal")
                    controller_result = outcome.get("result")
            else:
                phase_roles = _all_roles(phase)
                for role_index, item in enumerate(phase_roles):
                    role = item["role"]
                    job = self._job_for(phase["phase_id"], role)
                    if (
                        self.model == "Cylinder"
                        and phase["phase_id"] == "regression-1"
                        and role == "contained_probe_helper"
                        and self.phase_dispositions.get("accounting_controller_leaf") != "success"
                    ):
                        self._append_role_not_started(
                            phase["phase_id"],
                            role,
                            "contained probe requires a successful accounting_controller_leaf terminal",
                        )
                        for remaining in phase_roles[role_index + 1 :]:
                            remaining_role = remaining["role"]
                            if remaining_role not in self.phase_dispositions:
                                remaining_job = self._job_for(phase["phase_id"], remaining_role)
                                reason = (
                                    "unused capacity"
                                    if remaining_job["entry_path"] is None
                                    else "regression-1 accounting prerequisite gate stopped phase"
                                )
                                self._append_role_not_started(
                                    phase["phase_id"], remaining_role, reason
                                )
                        status = "failed"
                        break
                    if (
                        self.model == "Cylinder"
                        and phase["phase_id"] == "regression-1"
                        and role == "regression_worker"
                    ):
                        prerequisite_roles = (
                            "accounting_controller_leaf",
                            "contained_probe_helper",
                        )
                        missing = [
                            prerequisite
                            for prerequisite in prerequisite_roles
                            if self.phase_dispositions.get(prerequisite) != "success"
                        ]
                        if missing:
                            reason = (
                                "regression-1 prerequisite roles lacked exact success: "
                                + ", ".join(missing)
                            )
                            self._append_role_not_started(phase["phase_id"], role, reason)
                            for remaining in phase_roles[role_index + 1 :]:
                                remaining_role = remaining["role"]
                                if remaining_role in self.phase_dispositions:
                                    continue
                                remaining_job = self._job_for(phase["phase_id"], remaining_role)
                                remaining_reason = (
                                    "unused capacity"
                                    if remaining_job["entry_path"] is None
                                    else "regression-1 prerequisite gate stopped phase"
                                )
                                self._append_role_not_started(
                                    phase["phase_id"], remaining_role, remaining_reason
                                )
                            status = "failed"
                            break
                    if job["entry_path"] is None:
                        self._append_role_not_started(phase["phase_id"], role, "unused capacity")
                        if (
                            self.model == "Cylinder"
                            and phase["phase_id"] == "regression-1"
                            and role in {"accounting_controller_leaf", "contained_probe_helper"}
                        ):
                            reason = f"regression-1 prerequisite {role} was not started"
                            for remaining in phase_roles[role_index + 1 :]:
                                remaining_role = remaining["role"]
                                if remaining_role in self.phase_dispositions:
                                    continue
                                remaining_job = self._job_for(phase["phase_id"], remaining_role)
                                remaining_reason = (
                                    "unused capacity"
                                    if remaining_job["entry_path"] is None
                                    else reason
                                )
                                self._append_role_not_started(
                                    phase["phase_id"], remaining_role, remaining_reason
                                )
                            status = "failed"
                            break
                        continue
                    handle = self._start_child(phase["phase_id"], role)
                    if handle is None:
                        if (
                            self.model == "Cylinder"
                            and phase["phase_id"] == "regression-1"
                            and role in {"accounting_controller_leaf", "contained_probe_helper"}
                            and self.phase_dispositions.get(role) != "success"
                        ):
                            reason = (
                                f"regression-1 prerequisite {role} "
                                "did not finish with exact success"
                            )
                            for remaining in phase_roles[role_index + 1 :]:
                                remaining_role = remaining["role"]
                                if remaining_role in self.phase_dispositions:
                                    continue
                                remaining_job = self._job_for(phase["phase_id"], remaining_role)
                                remaining_reason = (
                                    "unused capacity"
                                    if remaining_job["entry_path"] is None
                                    else reason
                                )
                                self._append_role_not_started(
                                    phase["phase_id"], remaining_role, remaining_reason
                                )
                            status = "failed"
                            break
                        continue
                    self._wait_process(handle)
                    outcome = self._reap_handle(handle, "terminal")
                    if (
                        self.model == "Cylinder"
                        and phase["phase_id"] == "regression-1"
                        and role in {"accounting_controller_leaf", "contained_probe_helper"}
                        and outcome.get("status") != "success"
                    ):
                        reason = (
                            f"regression-1 prerequisite {role} "
                            "did not finish with exact success"
                        )
                        for remaining in phase_roles[role_index + 1 :]:
                            remaining_role = remaining["role"]
                            if remaining_role in self.phase_dispositions:
                                continue
                            remaining_job = self._job_for(phase["phase_id"], remaining_role)
                            remaining_reason = (
                                "unused capacity"
                                if remaining_job["entry_path"] is None
                                else reason
                            )
                            self._append_role_not_started(
                                phase["phase_id"], remaining_role, remaining_reason
                            )
                        status = "failed"
                        break
        except Exception as exc:
            if self.active_phase != phase["phase_id"]:
                raise
            status = "failed"
            self.failed_reason = f"phase {phase['phase_id']} stopped: {type(exc).__name__}: {exc}"
            if self.unbound_children:
                self.failed_reason += "; an actual Popen child lacks a verified launch binding"
                raise GovernorError("unbound Popen child makes phase cleanup unconfirmed") from exc
            self._abort_live_children()
            if self.phase_started:
                raise GovernorError("phase cleanup is unconfirmed; preserve the open epoch and stop") from exc

        if self.model == "A2" and phase["phase_id"] == "accounting" and status == "success":
            try:
                self._verify_test_framework_sources()
            except Exception as exc:
                status = "failed"
                self.failed_reason = f"accounting test framework source changed during phase: {type(exc).__name__}: {exc}"

        if self.model == "A2" and phase["phase_id"] == "timing":
            self.timing_reconciliation = self._reconstruct_timing_calls()
            if self.timing_reconciliation.get("status") != "reconciled":
                status = "failed"
                self.failed_reason = self.failed_reason or "fixed timing call reconciliation is invalid"

        if any(
            self.phase_dispositions.get(role) != "success"
            for role in self._required_roles_for_phase(phase)
        ):
            status = "failed"
        return self._finish_phase(phase, status, controller_result=controller_result)

    def _required_roles_for_phase(self, phase: dict[str, Any]) -> set[str]:
        phase_id = phase["phase_id"]
        if self.model == "A2":
            return {
                "environment": {"environment_leaf"},
                "accounting": {
                    "accounting_test_controller",
                    "accounting_timer_fixture",
                    "accounting_pdeath_refusal",
                    "accounting_handshake_leaf",
                    "accounting_cleanup_leaf",
                },
                "retained": {"retained_leaf"},
                "metadata": {"metadata_controller", "metadata_worker"},
                "timing": {"timing_controller", "timing_worker"},
                "terminal": {"record_leaf"},
            }[phase_id]
        if phase_id == "preflight":
            return {"environment_leaf"}
        if phase_id == "regression-1":
            return {"regression_worker", "accounting_controller_leaf", "contained_probe_helper"}
        if phase_id.startswith("regression-"):
            return {"regression_worker"}
        if phase_id.startswith("row-"):
            return {"row_worker"}
        if phase_id == "terminal":
            return {"record_leaf"}
        raise GovernorError(f"phase lacks a fixed required-role profile: {phase_id}")

    def _abort_live_children(self) -> None:
        for handle in list(self.phase_started.values()):
            proc = handle["process"]
            try:
                self._wait_process(handle, force_terminate=True)
            except (OSError, ProcfsError, GovernorError, subprocess.TimeoutExpired):
                self.cleanup_started = True
                continue
            if proc.poll() is not None and "terminal" not in handle:
                try:
                    self._reap_handle(handle, "failed")
                except Exception:
                    self.cleanup_started = True

    def _reconstruct_timing_calls(self) -> dict[str, Any]:
        """Reconcile the frozen 48 slots after reap within the existing timing phase."""
        report_path = self.output_root / "phases" / "timing" / "timing-reconciliation.json"
        report: dict[str, Any]
        raw_evidence = b""
        event_evidence: list[dict[str, Any]] = []
        expected_report_slots: list[dict[str, Any]] = []
        pilot_adapter_evidence: dict[str, Any] | None = None
        invalid_trailing_record: dict[str, Any] | None = None
        record_path: Path | None = None
        try:
            external = self.manifest.get("external_pins")
            pin = external.get("pilot_selection") if isinstance(external, dict) else None
            if not isinstance(pin, dict) or set(pin) != {"path", "sha256"}:
                raise GovernorError("fixed pilot selection pin is absent from the manifest")
            pilot_raw = _read_bytes(Path(pin["path"]), limit=8 * 1024 * 1024)
            map_pin = external.get("coverage_map") if isinstance(external, dict) else None
            if not isinstance(map_pin, dict) or set(map_pin) != {"path", "sha256"}:
                raise GovernorError("frozen coverage-map pin is absent from the manifest")
            map_raw = _read_bytes(Path(map_pin["path"]), limit=16 * 1024 * 1024)
            if hashlib.sha256(map_raw).hexdigest() != map_pin["sha256"]:
                raise GovernorError("frozen coverage-map bytes differ from the manifest pin")
            fixture_raw = _read_bytes(
                self.worktree_root / "docs/design/gen1/stochastic/A2-fixtures-v1.json",
                limit=8 * 1024 * 1024,
            )
            self._verify_source_binding()
            coverage_relative = "tests/oracles/gen1/stochastic_formal/a2/v2/coverage.py"
            coverage_path = self.worktree_root / coverage_relative
            coverage_source = _read_bytes(coverage_path, limit=2 * 1024 * 1024)
            if hashlib.sha256(coverage_source).hexdigest() != self.catalog["files"].get(coverage_relative):
                raise GovernorError("A2 coverage adapter bytes differ from the verified source catalog")
            coverage_namespace: dict[str, Any] = {
                "__file__": str(coverage_path),
                "__name__": "_gragra_a2_v2_coverage_for_timing_replay",
            }
            exec(compile(coverage_source, str(coverage_path), "exec"), coverage_namespace)

            fixture_map = coverage_namespace["build_map"](fixture_raw)
            map_evidence = coverage_namespace["verify_map_artifact"](
                fixture_map, map_raw, map_pin["sha256"]
            )

            pilot_entries, pilot_adapter_evidence = coverage_namespace[
                "adapt_frozen_pilot_selection"
            ](
                pilot_raw,
                fixture_map,
                coverage_map_sha256=map_evidence["sha256"],
                expected_sha256=pin["sha256"],
            )
            expected: dict[tuple[str, int], dict[str, Any]] = {}
            for entry in pilot_entries:
                for iteration in (0, 1, 2, 3):
                    key = (entry["canonical_key_json"], iteration)
                    if key in expected:
                        raise GovernorError("pilot repeats a native key/iteration slot")
                    expected[key] = {
                        "canonical_key_json": entry["canonical_key_json"],
                        "fixture_id": entry["native_key"].get("fixture_id"),
                        "record_kind": entry["record_kind"],
                        "representation": entry["native_key"].get("representation"),
                        "input_convention": (
                            "nominal_exact_x_scalar"
                            if entry["record_kind"] == "nominal_scalar"
                            else "frozen_physical_tuple"
                        ),
                        "iteration": iteration,
                        "native_key": entry["native_key"],
                    }
            if len(expected) != 48:
                raise GovernorError("pilot did not expand to exactly 48 native timing slots")
            expected_report_slots = list(expected.values())
            worker_handles = [
                item
                for item in self.all_handles.values()
                if item["phase"] == "timing" and item["role"] == "timing_worker"
            ]
            if len(worker_handles) > 1:
                raise GovernorError("timing worker role was launched more than once")
            worker = worker_handles[0] if worker_handles else None
            record_path = self.output_root / "phases" / "timing" / "timing-calls.jsonl"
            raw = _read_bytes(record_path, limit=8 * 1024 * 1024) if record_path.exists() else b""
            raw_evidence = raw
            if worker is not None and (
                "terminal" not in worker or "popen_reap_evidence" not in worker
            ):
                raise GovernorError("timing JSONL cannot be reconciled before verified worker reap")
            if worker is None and raw:
                raise GovernorError("timing JSONL exists without a declared timing worker")
            events = event_evidence
            slot_events: dict[tuple[str, int], dict[str, list[dict[str, Any]]]] = {
                key: {"call_start": [], "call_complete": [], "call_error": []} for key in expected
            }
            if raw:
                final_newline = raw.rfind(b"\n")
                complete_prefix = raw[: final_newline + 1] if final_newline >= 0 else b""
                trailing_bytes = raw[final_newline + 1 :] if final_newline + 1 < len(raw) else b""
                if trailing_bytes:
                    tail_path = record_path.with_name("timing-calls-invalid-trailing-bytes.bin")
                    _write_new(tail_path, trailing_bytes)
                    invalid_trailing_record = {
                        "status": "partial_unterminated_record_not_parsed",
                        "path": str(tail_path),
                        "sha256": hashlib.sha256(trailing_bytes).hexdigest(),
                        "byte_count": len(trailing_bytes),
                        "original_jsonl_preserved": True,
                        "original_jsonl_path": str(record_path),
                        "original_jsonl_sha256": hashlib.sha256(raw).hexdigest(),
                    }
                lines = complete_prefix.splitlines()
                if len(lines) > 96:
                    raise GovernorError("timing JSONL exceeds the fixed two-events-per-slot limit")
                for line_number, line in enumerate(lines, 1):
                    if not line or len(line) > 1_048_576:
                        raise GovernorError(f"timing JSONL line {line_number} is empty or too large")
                    event = strict_json(line)
                    if not isinstance(event, dict) or event.get("event") not in {
                        "call_start", "call_complete", "call_error"
                    }:
                        raise GovernorError(f"timing JSONL line {line_number} has an unknown event")
                    slot = (event.get("canonical_key_json"), event.get("iteration"))
                    if slot not in slot_events:
                        raise GovernorError(f"timing JSONL line {line_number} is outside the frozen 48 slots")
                    binding = expected[slot]
                    for name in (
                        "canonical_key_json", "fixture_id", "record_kind", "representation",
                        "input_convention", "iteration",
                    ):
                        if event.get(name) != binding[name]:
                            raise GovernorError(f"timing JSONL line {line_number} contradicts slot field {name}")
                    if event.get("same_process") is not True:
                        raise GovernorError(f"timing JSONL line {line_number} lacks same-process evidence")
                    if worker is None or (
                        event.get("pid") != worker["child_binding"]["pid"]
                        or event.get("starttime_ticks") != worker["child_binding"]["starttime_ticks"]
                    ):
                        raise GovernorError(f"timing JSONL line {line_number} is not bound to the reaped worker")
                    if worker["terminal_mono_ns"] is not None and event.get("recorded_monotonic_ns", event.get("ended_monotonic_ns", 0)) > worker["terminal_mono_ns"]:
                        raise GovernorError(f"timing JSONL line {line_number} is later than worker reap")
                    slot_events[slot][event["event"]].append(event)
                    events.append(event)
            slots = []
            for key, identity in expected.items():
                observed = slot_events[key]
                starts, completes, errors = (
                    observed["call_start"], observed["call_complete"], observed["call_error"]
                )
                if len(starts) > 1 or len(completes) > 1 or len(errors) > 1 or (completes and errors):
                    raise GovernorError("timing JSONL contains duplicate or contradictory slot events")
                if (completes or errors) and not starts:
                    raise GovernorError("timing terminal event has no durable call_start marker")
                if completes:
                    completion = completes[0]
                    elapsed_ns = completion.get("elapsed_ns")
                    if type(elapsed_ns) is not int or elapsed_ns < 0:
                        raise GovernorError("completed call elapsed_ns is malformed")
                    state = "completed"
                elif errors:
                    completion = None
                    error = errors[0]
                    if type(error.get("elapsed_ns")) is not int or error["elapsed_ns"] < 0:
                        raise GovernorError("errored call elapsed_ns is malformed")
                    state = "errored"
                elif starts:
                    completion = None
                    error = None
                    state = "censored_interrupted"
                else:
                    completion = None
                    error = None
                    state = "unmeasured_not_started"
                row = {**identity, "status": state, "call_start_event": starts[0] if starts else None}
                if state == "completed":
                    row["completed_event"] = completion
                elif state == "errored":
                    row["error_event"] = error
                elif state == "censored_interrupted":
                    row.update(
                        {
                            "interrupted_event": starts[0],
                            "api_elapsed_lower_bound_seconds": None,
                            "elapsed_claim": "not_claimable_from_pre_call_call_start_marker",
                        }
                    )
                slots.append(row)
            completed = [row["completed_event"] for row in slots if row["status"] == "completed"]

            def summarize(calls: list[dict[str, Any]]) -> dict[str, Any]:
                seconds = [call["elapsed_ns"] / 1_000_000_000 for call in calls]
                if not seconds:
                    return {"count": 0}
                return {
                    "count": len(seconds),
                    "mean_seconds": statistics.fmean(seconds),
                    "median_seconds": statistics.median(seconds),
                    "population_stdev_seconds": statistics.pstdev(seconds),
                    "max_seconds": max(seconds),
                }

            by_iteration = {
                name: [call for call in completed if (call["iteration"] == 0) == (name == "initial")]
                for name in ("initial", "warm")
            }
            kinds = {kind: [call for call in completed if call["record_kind"] == kind] for kind in ("nominal_scalar", "physical_csd")}
            native_keys = {
                key_text: [call for call in completed if call["canonical_key_json"] == key_text]
                for key_text in sorted({call["canonical_key_json"] for call in completed})
            }
            censored = [row for row in slots if row["status"] == "censored_interrupted"]
            unmeasured = [row for row in slots if row["status"] == "unmeasured_not_started"]
            errored = [row for row in slots if row["status"] == "errored"]
            worker_lifetime = None
            if worker is not None:
                end_ns = worker.get("terminal_mono_ns")
                worker_lifetime = {
                    "pid": worker["child_binding"]["pid"],
                    "starttime_ticks": worker["child_binding"]["starttime_ticks"],
                    "start_monotonic_ns": worker["start_mono_ns"],
                    "terminal_monotonic_ns": end_ns,
                    "elapsed_seconds": None if end_ns is None else max(0, end_ns - worker["start_mono_ns"]) / 1_000_000_000,
                    "scope": "whole_worker_lifetime_not_api_duration",
                }
            report = {
                "schema": "gragra.a2.diagnostic.timing-reconciliation.v1",
                "status": "invalid_trailing_partial_record" if invalid_trailing_record else "reconciled",
                "diagnostic_only": True,
                "science_credit": 0,
                "formal_comparison": "REJECTED",
                "expected_slot_count": 48,
                "observed_event_count": len(events),
                "timing_jsonl_path": str(record_path),
                "timing_jsonl_sha256": hashlib.sha256(raw).hexdigest() if record_path.exists() else None,
                "pilot_selection_sha256": pin["sha256"],
                "coverage_map_sha256": map_pin["sha256"],
                "pilot_adapter": pilot_adapter_evidence,
                "invalid_trailing_record": invalid_trailing_record,
                "counts": {
                    "completed": len(completed), "errored": len(errored),
                    "censored_interrupted": len(censored), "unmeasured_not_started": len(unmeasured),
                },
                "completed_call_summary": {
                    "all": summarize(completed),
                    "iteration": {name: summarize(calls) for name, calls in by_iteration.items()},
                    "record_kind": {name: summarize(calls) for name, calls in kinds.items()},
                    "native_key": {name: summarize(calls) for name, calls in native_keys.items()},
                },
                "completed_events": [row["completed_event"] for row in slots if row["status"] == "completed"],
                "error_events": [row["error_event"] for row in slots if row["status"] == "errored"],
                "events_in_file_order": events,
                "censored_slots": censored,
                "unmeasured_slots": unmeasured,
                "slots": slots,
                "worker_lifetime": worker_lifetime,
                "api_duration_claim_for_censored_slots": "none",
            }
        except Exception as exc:
            report = {
                "schema": "gragra.a2.diagnostic.timing-reconciliation.v1",
                "status": "invalid",
                "diagnostic_only": True,
                "science_credit": 0,
                "formal_comparison": "REJECTED",
                "expected_slot_count": 48,
                "error": f"{type(exc).__name__}: {exc}",
                "timing_jsonl_path": str(record_path) if record_path is not None else None,
                "timing_jsonl_sha256": hashlib.sha256(raw_evidence).hexdigest() if raw_evidence else None,
                "invalid_trailing_record": invalid_trailing_record,
                "observed_events_before_refusal": event_evidence,
                "slots": [
                    {**slot, "status": "reconciliation_invalid"}
                    for slot in expected_report_slots
                ],
                "api_duration_claim_for_censored_slots": "none",
            }
        _write_new(report_path, _canonical(report))
        report["report_path"] = str(report_path)
        report["report_sha256"] = sha256_file(report_path)
        return report

    def _finalize_root(self, phase_status: str) -> dict[str, Any]:
        self.cleanup_started = True
        self._close_unbound_children()
        self._abort_live_children()
        unknown_reaped = self._reap_unexpected_children()
        receipt = cleanup_inventory(
            root_pid=self.root["pid"],
            declared=self.declared,
            reaped=self.reaped,
        )
        cgroup_receipt = None
        if self.cgroup is not None:
            try:
                self.cgroup_sample = self.cgroup.sample()
                cgroup_receipt = self.cgroup.close()
            except (GovernorError, OSError) as exc:
                cgroup_receipt = {
                    "policy": "hard_cgroup_v2_memory.max",
                    "limit_bytes": _CgroupMemoryLimit.LIMIT_BYTES,
                    "confirmed": False,
                    "cleanup_error": f"{type(exc).__name__}: {exc}",
                }
            if cgroup_receipt.get("confirmed") is not True:
                receipt["confirmed"] = False
                receipt.setdefault("visibility_errors", []).append(
                    {"kind": "cgroup_cleanup_unconfirmed", "evidence": cgroup_receipt}
                )
        else:
            cgroup_receipt = {
                "policy": "soft_1s_procfs",
                "limit_bytes": _CgroupMemoryLimit.LIMIT_BYTES,
                "sample_interval_seconds": 1,
                "possible_overshoot": True,
                "confirmed": None,
            }
        if unknown_reaped:
            receipt["unknown_reaped_children"].extend(unknown_reaped)
            receipt["confirmed"] = False
        if self.unbound_child_evidence:
            receipt["confirmed"] = False
            receipt.setdefault("visibility_errors", []).append(
                {
                    "kind": "popen_child_without_verified_launch_binding",
                    "children": self.unbound_child_evidence,
                }
            )
        if any(item.get("cleanup_unconfirmed") is True for item in self.role_maximum_events):
            receipt["confirmed"] = False
            receipt.setdefault("visibility_errors", []).append(
                {
                    "kind": "role_maximum_wait_unconfirmed",
                    "events": self.role_maximum_events,
                }
            )
        if any(handle["process"].poll() is None for handle in self.phase_started.values()):
            receipt["confirmed"] = False
        release_evidence = self.host_admission.release(
            cleanup_confirmed=receipt.get("confirmed") is True
        )
        receipt["host_worker_admission"] = release_evidence
        if (
            self.host_admission.evidence.get("status") == "acquired"
            and release_evidence.get("status") != "released_after_confirmed_cleanup"
        ):
            release_evidence["cleanup_confirmed"] = False
            receipt["confirmed"] = False
            receipt.setdefault("visibility_errors", []).append(
                {"kind": "host_worker_admission_release_unconfirmed", "evidence": release_evidence}
            )
        final_status = (
            "unconfirmed_cleanup"
            if receipt.get("confirmed") is not True
            else "success"
            if phase_status == "success"
            else "failed"
        )
        terminal_mono_ns = time.monotonic_ns()
        elapsed_upper_ns = max(
            0,
            terminal_mono_ns - self.root_timer["root_start_mono_lower_ns"],
        )
        phase_reserved_seconds = sum(
            item["reserved_seconds"] for item in self.phase_reservations
        )
        child_measurements = []
        child_elapsed_total = 0.0
        child_elapsed_complete = not bool(self.unbound_child_evidence)
        for handle in self.all_handles.values():
            start_ns = handle["start_mono_ns"]
            end_ns = handle.get("terminal_mono_ns")
            elapsed = None if end_ns is None else max(0, end_ns - start_ns) / 1_000_000_000
            if elapsed is None:
                child_elapsed_complete = False
            else:
                child_elapsed_total += elapsed
            child_measurements.append(
                {
                    "phase_id": handle["phase"],
                    "role": handle["role"],
                    "pid": handle["child_binding"]["pid"],
                    "start_mono_ns": start_ns,
                    "terminal_mono_ns": end_ns,
                    "elapsed_seconds": elapsed,
                }
            )
        intent = {
            "schema": "gragra-diagnostic-root-terminal-intent-v1",
            "model": self.model,
            "epoch_id": self.epoch_id,
            "ledger_id": self.ledger_id,
            "status": final_status,
            "phase_status": phase_status,
            "root_identity": self.root,
            "cleanup_receipt": receipt,
            "host_worker_admission": self.host_admission.evidence,
            "timing_reconciliation": self.timing_reconciliation,
            "root_deadline_monotonic_ns": self.hard_deadline_ns,
            "soft_deadline_monotonic_ns": self.soft_deadline_ns,
            "root_terminal_monotonic_ns": terminal_mono_ns,
            "accounted_seconds": {
                "root_reserved": 900,
                "descendant_pre_authorized": 900,
                "descendant_phase_reserved": phase_reserved_seconds,
                "total_reserved": 900 + phase_reserved_seconds,
                "reserved_phases": self.phase_reservations,
                "unused_authorized_descendant_capacity": 900 - phase_reserved_seconds,
                "refund_seconds": 0,
                "old_credit_seconds": 0,
            },
            "observed_wall_seconds_upper_bound": elapsed_upper_ns / 1_000_000_000,
            "observed_wall_upper_bound_source": "terminal_monotonic_minus_conservative_root_start_lower_bound",
            "native_root_exit_receipt_required": True,
            "native_process_exit_confirmed": False,
            "child_elapsed_seconds": {
                "measurements": child_measurements,
                "sum_seconds": child_elapsed_total,
                "complete": child_elapsed_complete,
                "sum_is_wall_time": False,
            },
            "rss_policy": self.rss_policy,
            "rss_peak_bytes": self.rss_peak_bytes,
            "rss_sample_count": self.rss_sample_count,
            "cgroup_memory_limit": cgroup_receipt,
            "cgroup_last_sample": self.cgroup_sample,
            "failed_reason": self.failed_reason,
            "unbound_children": self.unbound_child_evidence,
            "role_maximum_events": self.role_maximum_events,
            "science_credit": 0,
            "old_credit_seconds": 0,
        }
        intent_path = self.output_root / "root-terminal-intent.json"
        _write_new(intent_path, _canonical(intent))
        if self.active_phase is None:
            self.ledger.append(
                "ROOT_TERMINAL",
                status=final_status,
                pid=self.root["pid"],
                starttime_ticks=self.root["starttime_ticks"],
                boot_id=self.root_boot_id,
                root_terminal_intent_schema=intent["schema"],
                cleanup_receipt=receipt,
                root_terminal_intent_sha256=sha256_file(intent_path),
                root_terminal_monotonic_ns=intent["root_terminal_monotonic_ns"],
                native_root_exit_receipt_required=True,
                native_process_exit_confirmed=False,
                host_worker_admission=self.host_admission.evidence,
                science_credit=0,
                old_credit_seconds=0,
            )
        self.ledger.close()
        return intent

    def run(self) -> dict[str, Any]:
        if self.host_admission.evidence.get("status") != "acquired":
            self.failed_reason = "manifest-bound host worker admission refused: " + str(
                self.host_admission.evidence.get("reason", "no slot acquired")
            )
            return self._finalize_root("failed")
        phase_status = "success"
        for phase in PHASE_SCHEDULE[self.model]:
            if time.monotonic_ns() >= self.soft_deadline_ns:
                self.failed_reason = "root soft deadline reached before the next phase reservation"
                phase_status = "failed"
                break
            try:
                result = self._execute_phase(phase)
            except Exception as exc:
                self.failed_reason = f"governor phase failure: {type(exc).__name__}: {exc}"
                phase_status = "failed"
                break
            if result != "success":
                phase_status = "failed"
                break
        intent = self._finalize_root(phase_status)
        return intent


def main(argv: list[str] | None = None) -> int:
    import argparse

    try:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--model", choices=("A2", "Cylinder"), required=True)
        parser.add_argument("--manifest", required=True)
        args = parser.parse_args(sys.argv[1:] if argv is None else argv)
        manifest_path = _safe_manifest_path(args.manifest, "manifest")
        governor = RootGovernor(args.model, manifest_path, _ROOT_TIMER_EVIDENCE)
        terminal = governor.run()
        return 0 if terminal.get("status") == "success" else 1
    except BaseException as exc:
        os.write(2, f"diagnostic root refused/failed: {type(exc).__name__}: {exc}\n".encode("utf-8", "replace"))
        return 127


if __name__ == "__main__":
    os._exit(main())
