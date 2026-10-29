"""Linux process identities and cleanup evidence for the shared diagnostic root."""

from __future__ import annotations

import ctypes
import errno
import os
from pathlib import Path
from typing import Any


class ProcfsError(RuntimeError):
    """Process identity or cleanup evidence could not be reconciled."""


class ProcfsGone(ProcfsError):
    """A PID vanished between inventory enumeration and identity read."""


def _raise_proc_error(message: str, exc: OSError) -> None:
    if exc.errno in {errno.ENOENT, errno.ESRCH}:
        raise ProcfsGone(message) from exc
    raise ProcfsError(f"{message}: errno={exc.errno}") from exc


def _stat_fields(raw: bytes) -> tuple[int, list[bytes]]:
    prefix, separator, tail = raw.rpartition(b")")
    if not separator:
        raise ProcfsError("proc stat lacks its final comm delimiter")
    pid_field = prefix.split(maxsplit=1)[0]
    if not pid_field.isdigit():
        raise ProcfsError("proc stat PID field is malformed")
    return int(pid_field), tail.strip().split()


def _stat_identity(pid: int, fields: list[bytes]) -> dict[str, int]:
    if pid <= 0 or len(fields) < 22:
        raise ProcfsError("proc stat PID or field count mismatch")
    try:
        identity = {
            "pid": pid,
            "ppid": int(fields[1]),
            "pgrp": int(fields[2]),
            "session": int(fields[3]),
            "starttime_ticks": int(fields[19]),
        }
    except ValueError as exc:
        raise ProcfsError("proc stat identity fields are malformed") from exc
    if identity["starttime_ticks"] <= 0 or any(
        identity[field] < 0 for field in ("ppid", "pgrp", "session")
    ):
        raise ProcfsError("proc stat identity values are invalid")
    return identity


def validate_identity_bracket(before: dict[str, int], after: dict[str, int]) -> None:
    """Require stat/status/argv sampling to stay on one PID lifetime and parent."""
    fields = ("pid", "starttime_ticks", "ppid", "pgrp", "session")
    for record in (before, after):
        if not isinstance(record, dict):
            raise ProcfsError("stat bracket identity must be an object")
        for field in fields:
            value = record.get(field)
            if type(value) is not int or value < (1 if field in {"pid", "starttime_ticks"} else 0):
                raise ProcfsError(f"stat bracket {field} is invalid")
    if any(before[field] != after[field] for field in fields):
        raise ProcfsError("process PID/starttime/parent/group/session changed during identity read")


def boot_id() -> str:
    try:
        value = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    except OSError as exc:
        raise ProcfsError("kernel boot ID is unavailable") from exc
    if len(value) != 36 or value.count("-") != 4:
        raise ProcfsError("kernel boot ID is malformed")
    return value


def parse_cmdline(raw: bytes) -> list[str]:
    """Decode proc cmdline, accepting an empty zombie/kernel-process argv."""
    if raw == b"":
        return []
    if not raw.endswith(b"\0"):
        raise ProcfsError("process argv lacks its terminal NUL")
    return [os.fsdecode(part) for part in raw[:-1].split(b"\0")]


def process_argv(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError as exc:
        _raise_proc_error(f"process argv unavailable for PID {pid}", exc)
    return parse_cmdline(raw)


def process_identity(pid: int, *, require_argv: bool = False) -> dict[str, Any]:
    """Read PID/starttime/PPID/group/session/RSS/argv as one checkable identity."""
    if type(pid) is not int or pid <= 0:
        raise ProcfsError("PID must be a positive integer")
    proc_fd = -1
    try:
        proc_fd = os.open(
            f"/proc/{pid}",
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
        )
    except OSError as exc:
        _raise_proc_error(f"process directory unavailable for PID {pid}", exc)

    def read_at(name: str) -> bytes:
        fd = -1
        try:
            fd = os.open(name, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0), dir_fd=proc_fd)
            chunks = []
            while True:
                chunk = os.read(fd, 65536)
                if not chunk:
                    return b"".join(chunks)
                chunks.append(chunk)
        except OSError as exc:
            _raise_proc_error(f"process {name} unavailable for PID {pid}", exc)
        finally:
            if fd >= 0:
                os.close(fd)

    try:
        initial_stat = read_at("stat")
        initial_pid, initial_fields = _stat_fields(initial_stat)
        if initial_pid != pid:
            raise ProcfsError("initial proc stat PID or field count mismatch")
        initial_identity = _stat_identity(initial_pid, initial_fields)

        status_raw = read_at("status")
        try:
            status = status_raw.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ProcfsError("process status is not ASCII") from exc
        argv: list[str] | None
        argv_error: int | None = None
        try:
            argv = parse_cmdline(read_at("cmdline"))
        except ProcfsGone:
            raise
        except ProcfsError as exc:
            if require_argv:
                raise
            argv = None
            marker = "errno="
            message = str(exc)
            argv_error = int(message.split(marker, 1)[1].split()[0]) if marker in message else None

        final_stat = read_at("stat")
        final_pid, final_fields = _stat_fields(final_stat)
        if final_pid != pid:
            raise ProcfsError("final proc stat PID or field count mismatch")
        try:
            final_identity = _stat_identity(final_pid, final_fields)
            state = final_fields[0].decode("ascii")
            rss_pages = int(final_fields[21])
        except (UnicodeDecodeError, ValueError, ProcfsError) as exc:
            if isinstance(exc, ProcfsError):
                raise
            raise ProcfsError("final proc stat identity fields are malformed") from exc
        validate_identity_bracket(initial_identity, final_identity)
        if rss_pages < 0:
            raise ProcfsError("proc RSS field is invalid")

        flags: dict[str, str] = {}
        for line in status.splitlines():
            name, sep, value = line.partition(":")
            if sep and name in {"NoNewPrivs", "Seccomp", "Threads"}:
                flags[name] = value.strip()
        if "Threads" not in flags:
            raise ProcfsError("thread count is absent from process status")
        try:
            rss_bytes = rss_pages * int(os.sysconf("SC_PAGE_SIZE"))
            threads = int(flags["Threads"])
        except (OSError, ValueError) as exc:
            raise ProcfsError("RSS or thread count cannot be interpreted") from exc
        return {
            "pid": pid,
            "ppid": final_identity["ppid"],
            "pgrp": final_identity["pgrp"],
            "session": final_identity["session"],
            "starttime_ticks": final_identity["starttime_ticks"],
            "state": state,
            "argv": argv,
            "argv_error": argv_error,
            "boot_id": boot_id(),
            "rss_bytes": rss_bytes,
            "threads": threads,
            "NoNewPrivs": flags.get("NoNewPrivs"),
            "Seccomp": flags.get("Seccomp"),
        }
    finally:
        os.close(proc_fd)


def root_identity() -> dict[str, Any]:
    pid = os.getpid()
    identity = process_identity(pid, require_argv=True)
    if identity["threads"] != 1:
        raise ProcfsError("governor root must remain single-threaded")
    identity["expected_argv"] = process_argv(pid)
    return identity


def root_identity_matches(expected: dict[str, Any]) -> bool:
    try:
        actual = process_identity(int(expected["pid"]), require_argv=True)
    except (ProcfsError, KeyError, TypeError, ValueError):
        return False
    return (
        actual["boot_id"] == expected.get("boot_id")
        and actual["starttime_ticks"] == expected.get("starttime_ticks")
        and actual["argv"] == expected.get("argv", expected.get("expected_argv"))
    )


def task_ids(pid: int) -> list[int]:
    try:
        values = sorted(int(path.name) for path in Path(f"/proc/{pid}/task").iterdir())
    except (OSError, ValueError) as exc:
        raise ProcfsError(f"task inventory unavailable for PID {pid}") from exc
    return values


def enumerate_processes() -> list[dict[str, Any]]:
    """Return visible process identities; only ENOENT/ESRCH races are omitted."""
    results: list[dict[str, Any]] = []
    try:
        pids = sorted(int(path.name) for path in Path("/proc").iterdir() if path.name.isdigit())
    except (OSError, ValueError) as exc:
        raise ProcfsError("proc PID inventory is unavailable") from exc
    for pid in pids:
        try:
            results.append(process_identity(pid, require_argv=False))
        except ProcfsGone:
            # Genuine exit races are reconciled against task children and waitpid.
            continue
        except ProcfsError:
            # Permission, parse, status, and identity errors must fail cleanup
            # closed instead of making an unknown process disappear.
            raise
    return results


def classify_proc_scan_error(error: ProcfsError) -> str:
    """Only a typed ENOENT/ESRCH race is an ordinary disappearance."""
    return "vanished" if isinstance(error, ProcfsGone) else "unobserved"


def process_group_members(pgid: int) -> list[dict[str, Any]]:
    if type(pgid) is not int or pgid <= 0:
        raise ProcfsError("process-group id is invalid")
    return [identity for identity in enumerate_processes() if identity["pgrp"] == pgid]


def child_identities(root_pid: int) -> list[dict[str, Any]]:
    return [identity for identity in enumerate_processes() if identity["ppid"] == root_pid]


def direct_child_pids(pid: int) -> list[int]:
    """Read kernel task child lists so zombies are visible before a wait."""
    tids = task_ids(pid)
    found: set[int] = set()
    for tid in tids:
        path = Path(f"/proc/{pid}/task/{tid}/children")
        try:
            raw = path.read_bytes()
        except OSError as exc:
            _raise_proc_error(f"task child list unavailable for TID {tid}", exc)
        try:
            pieces = raw.decode("ascii").split()
            children = [int(piece) for piece in pieces]
        except (UnicodeDecodeError, ValueError) as exc:
            raise ProcfsError(f"task child list is malformed for TID {tid}") from exc
        if any(child <= 0 for child in children):
            raise ProcfsError("task child list contains a nonpositive PID")
        found.update(children)
    return sorted(found)


def reap_untracked_waitables(known_pids: set[int]) -> list[dict[str, Any]]:
    """Drain exited adopted/unknown children after all declared Popen waits."""
    unexpected: list[dict[str, Any]] = []
    while True:
        try:
            pid, status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            break
        except OSError as exc:
            if exc.errno == errno.ECHILD:
                break
            raise ProcfsError(f"waitpid(-1, WNOHANG) failed: errno={exc.errno}") from exc
        if pid == 0:
            break
        if pid not in known_pids:
            unexpected.append(
                {
                    "pid": pid,
                    "wait_status": status,
                    "identity_known": False,
                    "reason": "unregistered_child_reaped",
                }
            )
        else:
            unexpected.append(
                {
                    "pid": pid,
                    "wait_status": status,
                    "identity_known": False,
                    "reason": "declared_child_reaped_outside_its_popen_wait",
                }
            )
    return unexpected


def set_child_subreaper() -> dict[str, int]:
    """Enable and read back PR_SET_CHILD_SUBREAPER on the root."""
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        prctl = libc.prctl
    except AttributeError as exc:
        raise ProcfsError("libc.prctl is unavailable for subreaper setup") from exc
    prctl.restype = ctypes.c_int
    prctl.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    ctypes.set_errno(0)
    if prctl(36, ctypes.c_ulong(1), ctypes.c_ulong(0), ctypes.c_ulong(0), ctypes.c_ulong(0)) != 0:
        code = ctypes.get_errno()
        raise ProcfsError(f"PR_SET_CHILD_SUBREAPER failed with errno {code}")
    value = ctypes.c_int(0)
    address = ctypes.c_ulong(ctypes.addressof(value))
    ctypes.set_errno(0)
    if prctl(37, address, ctypes.c_ulong(0), ctypes.c_ulong(0), ctypes.c_ulong(0)) != 0:
        code = ctypes.get_errno()
        raise ProcfsError(f"PR_GET_CHILD_SUBREAPER failed with errno {code}")
    if value.value != 1:
        raise ProcfsError("PR_GET_CHILD_SUBREAPER did not read back enabled")
    return {"set": 1, "readback": int(value.value)}


def verify_no_fork_refusal() -> dict[str, Any]:
    """Expected negative control: a fork syscall must be refused with EPERM."""
    before = task_ids(os.getpid())
    try:
        os.fork()
    except OSError as exc:
        after = task_ids(os.getpid())
        if exc.errno != errno.EPERM or before != after:
            raise ProcfsError("no-fork negative control did not fail with EPERM/no child") from exc
        return {
            "attempt": "os.fork",
            "expected_errno": errno.EPERM,
            "observed_errno": exc.errno,
            "parent_task_ids_before": before,
            "parent_task_ids_after": after,
            "child_created": False,
        }
    else:
        # If seccomp is absent, stop immediately. This branch is unsafe to
        # continue in; the forked process exits without importing or writing.
        os._exit(125)


def cleanup_inventory(
    *,
    root_pid: int,
    declared: list[dict[str, Any]],
    reaped: list[dict[str, Any]],
) -> dict[str, Any]:
    """Conservatively reconcile known children and escaped process groups."""
    live = enumerate_processes()
    root_children = [item for item in live if item["ppid"] == root_pid]
    task_child_pid_list = direct_child_pids(root_pid)
    visibility_errors = []
    if any(item.get("argv") is None for item in live):
        visibility_errors.append(
            {
                "kind": "process_argv_unavailable",
                "pids": sorted(item["pid"] for item in live if item.get("argv") is None),
            }
        )
    expected_groups = {
        item["pgrp"]
        for item in declared
        if type(item.get("pgrp")) is int and item.get("pgrp") > 0
    }
    nonempty_groups = []
    for pgid in sorted(expected_groups):
        members = process_group_members(pgid)
        if members:
            nonempty_groups.append({"pgrp": pgid, "members": members})
    receipt = reconcile_cleanup_evidence(
        root_pid=root_pid,
        live_identities=live,
        task_child_pids=task_child_pid_list,
        declared=declared,
        reaped=reaped,
        nonempty_process_groups=nonempty_groups,
        visibility_errors=visibility_errors,
    )
    receipt["visibility_limit"] = (
        "limited to identities exposed by this process /proc mount; any visible identity "
        "error is unconfirmed, and hidden unrelated host processes are not targeted"
    )
    return receipt


def reconcile_cleanup_evidence(
    *,
    root_pid: int,
    live_identities: list[dict[str, Any]],
    task_child_pids: list[int],
    declared: list[dict[str, Any]],
    reaped: list[dict[str, Any]],
    nonempty_process_groups: list[dict[str, Any]],
    visibility_errors: list[dict[str, Any]],
) -> dict[str, Any]:
    """Pure identity reconciliation for deterministic charged accounting fixtures."""
    declared_by_identity = {
        (item.get("pid"), item.get("starttime_ticks")): item
        for item in declared
        if type(item.get("pid")) is int
    }
    reaped_by_identity = {
        (item.get("pid"), item.get("starttime_ticks")): item for item in reaped
    }
    proc_children = [item for item in live_identities if item.get("ppid") == root_pid]
    child_pids = {item.get("pid") for item in proc_children}
    task_pids = set(task_child_pids)
    if task_pids != child_pids:
        visibility_errors = [
            *visibility_errors,
            {
                "kind": "task_children_proc_inventory_mismatch",
                "task_children": sorted(task_pids),
                "proc_children": sorted(child_pids),
            },
        ]
    declared_live = []
    unknown_live = []
    for identity in proc_children:
        marker = (identity.get("pid"), identity.get("starttime_ticks"))
        if marker in declared_by_identity:
            declared_live.append(identity)
        else:
            unknown_live.append(identity)
    unreaped = [
        item
        for item in declared
        if type(item.get("pid")) is int
        and (item.get("pid"), item.get("starttime_ticks")) not in reaped_by_identity
    ]
    unknown_reaped = [
        item
        for item in reaped
        if (item.get("pid"), item.get("starttime_ticks")) not in declared_by_identity
    ]
    return {
        "unknown_pids": sorted(
            int(item["pid"]) for item in unknown_live if type(item.get("pid")) is int
        ),
        "live_identities": declared_live + unknown_live,
        "nonempty_process_groups": list(nonempty_process_groups),
        "unreaped_children": unreaped,
        "unknown_reaped_children": unknown_reaped,
        "visibility_errors": list(visibility_errors),
        "task_children_pids": sorted(task_pids),
        "reaped_identities": list(reaped),
        "confirmed": not (
            declared_live
            or unknown_live
            or nonempty_process_groups
            or unreaped
            or unknown_reaped
            or visibility_errors
        ),
    }
